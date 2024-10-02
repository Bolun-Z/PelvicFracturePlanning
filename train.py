import os
# Allow duplicate libraries for KMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Set visible CUDA devices (GPU 0 and 1)
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
# Set CUDA launch blocking for debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
from tqdm import tqdm
from torch import nn, optim
from utils.utils import load_dataset, save_images
from utils.utils import Grad
from utils.callbacks import LossHistory
from torch.utils.data import DataLoader
from utils.training_set import set_optimizer_lr, get_lr_scheduler
from utils.CreatDataset import CreatDataset
from torch.utils.tensorboard import SummaryWriter

from networks.Structural_Restoration_Network import Deformation_Net, Discriminator

from torch.cuda.amp import GradScaler
from argparse import ArgumentParser

# =========================Train parameters=======================
torch.manual_seed(100)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = ArgumentParser(description='Set training parameters.')
# Set various training parameters like batch size, epochs, and more
parser.add_argument('--batch_size', default=2, type=int, help='The batch size of training samples')
parser.add_argument('--epochs', default=200, type=int, help='The number of training epochs')
parser.add_argument('--Cuda', default=True, type=bool, help='Use CUDA for GPU acceleration')
parser.add_argument('--Multi_GPU', default=True, type=bool, help='Use multiple GPUs for training')
parser.add_argument('--AMP', default=True, type=bool, help='Use Automatic Mixed Precision for training')
parser.add_argument('--save_period', default=5, type=int, help='How often to save model checkpoints (in epochs)')
parser.add_argument('--model', default='Deformation_cycle', type=str, help='Which model to use for training')

parser.add_argument('--Init_lr', default=1e-3, type=float, help='Initial learning rate')
parser.add_argument('--Min_lr', default=1e-6, type=float, help='Minimum learning rate')
parser.add_argument('--lr_decay_type', default='cos', type=str, help='Type of learning rate decay (cosine or step)')
parser.add_argument('--input_shape', default=[144, 144, 144], type=int, help='Input image dimensions (depth, height, width)')
parser.add_argument('--Dataset_path', default='../image', type=str, help='Path to the dataset root')
parser.add_argument('--pretrained_model', default='', type=str, help='Path to a pretrained model, if any')
parser.add_argument('--save_dir', default='logs/', type=str, help='Directory to save training history and logs')
parser.add_argument('--output_path', default='logs/', type=str, help='Path to save output logs')
parser.add_argument('--optimizer_type', default='adam', type=str, help='Optimizer type (adam or sgd)')
parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for the optimizer')
parser.add_argument('--weight_decay', default=0, type=float, help='Weight decay for regularization (0 when using Adam)')

args = parser.parse_args()

def define_model():
    # Define and initialize the Deformation network and Discriminator
    if args.model == 'Deformation_cycle':
        model = Deformation_Net(input_shape=args.input_shape)
        discriminator = Discriminator(input_shape=args.input_shape)
        # Use multiple GPUs if enabled
        if args.Multi_GPU:
            model = nn.DataParallel(model)
            discriminator = nn.DataParallel(discriminator)
        model.train()
        model.to(device)

        discriminator.train()
        discriminator.to(device)

        return model, discriminator

def get_dataloader():
    # Load the dataset and prepare dataloaders for training and validation
    train_ratio = 0.9
    train_lines, val_lines = load_dataset(args.Dataset_path, train_ratio)
    num_train = len(train_lines)
    num_val = len(val_lines)

    train_dataset = CreatDataset(train_lines, args.input_shape, mode='Train')
    val_dataset = CreatDataset(val_lines, args.input_shape, mode='Train')

    trainloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=2, pin_memory=True, drop_last=True)
    valloader = DataLoader(val_dataset, shuffle=True, batch_size=args.batch_size, num_workers=2, pin_memory=True, drop_last=True)

    epoch_step = num_train // args.batch_size
    epoch_step_val = num_val // args.batch_size

    return trainloader, valloader, epoch_step, epoch_step_val

def train_eval(trainloader, valloader, epoch_step, epoch_step_val):
    # Initialize Tensorboard writer
    writer = SummaryWriter(log_dir=args.save_dir)
    # Define the model and discriminator
    model, discriminator = define_model()

    # Load pretrained model weights if provided
    if args.pretrained_model != '':
        print(f'Load weights {args.pretrained_model}.')
        pretrained_dict = torch.load(args.pretrained_model, map_location=device)
        model.load_state_dict(pretrained_dict, strict=False)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    print(f'{total_params / (1024 * 1024):.2f}M total parameters.')

    # =========================train strategy=======================
    # Define loss functions
    MSE_loss = nn.MSELoss()
    L1_loss = nn.L1Loss()
    Grad_loss = Grad()

    # Initialize gradient scaler for mixed precision training
    if args.AMP:
        scaler = GradScaler()

    # Adjust learning rate based on batch size
    nbs = 16
    lr_limit_max = 1e-4 if args.optimizer_type == 'adam' else 1e-1
    lr_limit_min = 1e-4 if args.optimizer_type == 'adam' else 5e-4
    Init_lr_fit = min(max(args.batch_size / nbs * args.Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(args.batch_size / nbs * args.Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    # Choose optimizer for model and discriminator
    optimizer = {
        'adam': optim.Adam(model.parameters(), Init_lr_fit, betas=(args.momentum, 0.999), weight_decay=args.weight_decay, eps=1e-3),
        'sgd': optim.SGD(model.parameters(), Init_lr_fit, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay)
    }[args.optimizer_type]

    optimizer_D = {
        'adam': optim.Adam(discriminator.parameters(), Init_lr_fit, betas=(args.momentum, 0.999), weight_decay=args.weight_decay, eps=1e-3),
        'sgd': optim.SGD(discriminator.parameters(), Init_lr_fit, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay)
    }[args.optimizer_type]

    # Record the loss history during training
    loss_history = LossHistory(args.save_dir, model, input_shape=args.input_shape)

    # Training loop
    for epoch in range(args.epochs):
        train_loss = 0
        val_loss = 0
        train_losses = []
        val_losses = []

        train_loss_p = 0
        discriminator_loss_p = 0

        # Get learning rate scheduler
        lr_scheduler_func = get_lr_scheduler(args.lr_decay_type, Init_lr_fit, Min_lr_fit, args.epochs)

        # =========================train=======================
        model.train()
        for iteration, batch in enumerate(tqdm(trainloader)):
            simulation_image, healthy_image = batch
            with torch.no_grad():
                healthy_image = healthy_image.to(device)
                simulation_image = simulation_image.to(device)

            optimizer.zero_grad()
            optimizer_D.zero_grad()

            real_label = torch.autograd.Variable(torch.ones(simulation_image.size(0), 1)).cuda()
            fake_label = torch.autograd.Variable(torch.zeros(simulation_image.size(0), 1)).cuda()

            # ============== Train Generator =================
            outputs, pos_flow = model(simulation_image)
            l1_loss = L1_loss(outputs, healthy_image)

            loss_grad = Grad_loss.loss(pos_flow)

            # GAN loss
            loss_GAN_AB = MSE_loss(discriminator(outputs.data), real_label)
            loss_value = loss_grad + l1_loss + 2 * loss_GAN_AB

            scaler.scale(loss_value).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss_value.item()

            # ============== Train Discriminator =================
            real_out = discriminator(healthy_image)
            loss_real_D = MSE_loss(real_out, real_label)

            fake_out = discriminator(simulation_image)
            loss_fake_D = MSE_loss(fake_out, fake_label)
            loss_D = loss_real_D + loss_fake_D

            scaler.scale(loss_D).backward()
            scaler.step(optimizer_D)
            scaler.update()

            discriminator_loss_p += loss_D.item()

            # =========================plot==========================
            # Save images for visualization
            with torch.no_grad():
                if iteration % 5 == 0:
                    healthy_image = healthy_image[0][0].byte()
                    healthy_image_array = healthy_image.cpu().numpy()
                    save_images(healthy_image_array, size=[16, 10], path='img//' + 'image.jpg')

                    pred = outputs[0][0].clone()
                    pred = pred.cpu().detach().numpy()
                    save_images(pred, size=[16, 10], path='img//' + 'pred.jpg')

        if args.model == 'Deformation_cycle':
            print(f'Training: Epoch {epoch} Average Generator loss: {train_loss_p / (iteration + 1):.4f}')
            print(f'Training: Epoch {epoch} Average Discriminator loss: {discriminator_loss_p / (iteration + 1):.4f}')

        train_losses.append(train_loss_p)

        # =========================val=======================
        model.eval()
        for iteration, batch in enumerate(tqdm(valloader)):
            simulation_image, healthy_image = batch
            with torch.no_grad():
                healthy_image = healthy_image.to(device)
                simulation_image = simulation_image.to(device)

                real_label = torch.autograd.Variable(torch.ones(simulation_image.size(0), 1)).cuda()

                outputs, pos_flow = model(simulation_image)

                l1_loss = L1_loss(outputs, healthy_image)
                loss_grad = Grad_loss.loss(pos_flow)

                loss_GAN_AB = MSE_loss(discriminator(outputs.data), real_label)
                loss_value_p = loss_grad + l1_loss + 2 * loss_GAN_AB

                val_loss += loss_value_p.item()
                val_losses.append(val_loss)

        loss_history.append_loss(epoch + 1, train_loss_p / epoch_step, val_loss / epoch_step_val)
        print(f'Validation: Epoch {epoch} Average loss: {val_loss / (iteration + 1):.4f}')

        # =========================save model=====================
        # Save model at periodic checkpoints
        if ((epoch + 1) % args.save_period == 0):
            SAVE_PATH = args.output_path + str(epoch + 1) + ".pth"
            print("Save path:", SAVE_PATH)
            torch.save(model.state_dict(), SAVE_PATH)

    # =========================Finish training=======================
    # Save the final trained model
    SAVE_PATH = args.output_path + "Final_model.pth"
    torch.save(model.state_dict(), SAVE_PATH)

    print("Training finished.")
    loss_history.writer.close()

if __name__ == '__main__':
    # Get data loaders and start training
    trainloader, valloader, epoch_step, epoch_step_val = get_dataloader()
    train_eval(trainloader, valloader, epoch_step, epoch_step_val)
