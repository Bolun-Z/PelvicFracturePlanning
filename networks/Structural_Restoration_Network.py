import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch.nn.functional as F
from torch.distributions.normal import Normal
from networks.Resnet3D import generate_ResNet
from monai.networks.nets import SwinUNETR

import numpy as np

class FeatExtractor(nn.Module):
    def __init__(self, in_channels, out_channels, kSize=(3, 3), stride=(1, 1), padding=(1, 1, 1), bias=True,
                 norm='InstanceNorm', activation='LeakReLU'):
        """
        Extracts 2D or 3D features from the input feature map.

        :param in_channels: Number of channels in the input feature map
        :param out_channels: Number of channels in the output feature map
        :param kSize: Kernel size for the convolutional layer (2D or 3D)
        :param stride: Stride size for the convolution
        :param padding: Padding for the convolution
        :param bias: Whether to include bias in the convolution
        :param norm: Normalization method (default is InstanceNorm)
        :param activation: Activation method (default is LeakyReLU)
        """
        super().__init__()
        # Set up convolutional layers
        self.conv_1 = nn.Conv3d(in_channels, out_channels, kernel_size=kSize[0], stride=stride[0], padding=padding, bias=bias)
        self.conv_2 = nn.Conv3d(out_channels, out_channels, kernel_size=kSize[1], stride=stride[1], padding=padding, bias=bias)

        # Set up normalization layers
        if norm == 'InstanceNorm':
            self.norm_1 = nn.InstanceNorm3d(out_channels, affine=True)
            self.norm_2 = nn.InstanceNorm3d(out_channels, affine=True)
        else:
            self.norm_1 = nn.BatchNorm3d(out_channels)
            self.norm_2 = nn.BatchNorm3d(out_channels)

        # Set up activation layers
        if activation == 'LeakReLU':
            self.activation_1 = nn.LeakyReLU(negative_slope=1e-2, inplace=True)
            self.activation_2 = nn.LeakyReLU(negative_slope=1e-2, inplace=True)
        else:
            self.activation_1 = nn.ReLU(inplace=True)
            self.activation_2 = nn.ReLU(inplace=True)

    def forward(self, x):
        # Forward pass through convolution, normalization, and activation
        x = self.conv_1(x)
        x = self.norm_1(x)
        x = self.activation_1(x)
        x = self.conv_2(x)
        x = self.norm_2(x)
        x = self.activation_2(x)
        return x


class Info_Inter_Module(nn.Module):
    def __init__(self, channel, M=2, k_size=3):
        """
        Local cross-channel information interaction attention mechanism for multi-dimensional feature fusion.

        :param channel: Number of input feature map channels
        :param M: Number of input features
        :param k_size: Kernel size for 1D convolution, determining the scale of information interaction
        """
        super().__init__()
        self.M = M
        self.channel = channel
        self.gap = torch.nn.AdaptiveAvgPool3d((1, 1, 1))

        # Create a list of convolutional layers
        self.convs = nn.ModuleList([])
        for i in range(self.M):
            self.convs.append(
                nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
            )

        # Softmax layer for attention mechanism
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        # Concatenate input feature maps and calculate attention
        batch_size, channel, _, _, _ = x1.shape
        feats = torch.cat([x1, x2], dim=1)
        feats = feats.view(batch_size, self.M, self.channel, feats.shape[2], feats.shape[3], feats.shape[4])

        # Summation and global average pooling
        feats_S = torch.sum(feats, dim=1)
        feats_G = self.gap(feats_S)
        feats_G = feats_G.squeeze(-1).squeeze(-1).transpose(-1, -2)

        # Apply attention mechanism
        attention_vectors = [conv(feats_G).transpose(-1, -2).unsqueeze(-1).unsqueeze(-1) for conv in self.convs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.M, channel, 1, 1, 1)
        attention_vectors = self.softmax(attention_vectors)

        # Apply attention to the feature maps
        feats_o = torch.sum(feats * attention_vectors.expand_as(feats), dim=1)

        return feats_o


class Layer_Down(nn.Module):
    def __init__(self, in_channels, out_channels, min_z=8, downsample=True):
        """
        Basic module for downsampling in the network.

        :param in_channels: Number of channels in the input feature map
        :param out_channels: Number of channels in the output feature map
        :param min_z: Minimum z-axis size to apply max-pooling along z-axis
        :param downsample: Whether to downsample the input
        """
        super().__init__()
        self.min_z = min_z
        self.downsample = downsample

        # Define 2D and 3D feature extractors
        self.Feat_extractor_2D = FeatExtractor(in_channels=in_channels, out_channels=out_channels,
                                               kSize=((1, 3, 3), (1, 3, 3)), stride=(1, 1), padding=(0, 1, 1))
        self.Feat_extractor_3D = FeatExtractor(in_channels=in_channels, out_channels=out_channels,
                                               kSize=(3, 3), stride=(1, 1), padding=(1, 1, 1))
        self.IIM = Info_Inter_Module(channel=out_channels)

    def forward(self, x):
        # Downsample if enabled, adjusting pooling depending on z-dimension
        if self.downsample:
            if x.shape[2] >= self.min_z:
                x = F.max_pool3d(x, kernel_size=(2, 2, 2), stride=(2, 2, 2))
            else:
                x = F.max_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # Perform feature extraction and interaction
        x = self.IIM(self.Feat_extractor_2D(x), self.Feat_extractor_3D(x))
        return x


class Layer_Up(nn.Module):
    def __init__(self, in_channels, out_channels, SKIP=True):
        """
        Basic module for upsampling in the network.

        :param in_channels: Number of input feature map channels
        :param out_channels: Number of output feature map channels
        :param SKIP: Whether to use skip connections
        """
        super().__init__()
        self.SKIP = SKIP

        # Define 2D and 3D feature extractors
        self.Feat_extractor_2D = FeatExtractor(in_channels=in_channels, out_channels=out_channels,
                                               kSize=((1, 3, 3), (1, 3, 3)), stride=(1, 1), padding=(0, 1, 1))
        self.Feat_extractor_3D = FeatExtractor(in_channels=in_channels, out_channels=out_channels,
                                               kSize=(3, 3), stride=(1, 1), padding=(1, 1, 1))
        self.IIM = Info_Inter_Module(channel=out_channels)

    def forward(self, x):
        # If skip connections are enabled, concatenate input and skip features
        if self.SKIP:
            x, xskip = x
            tarSize = xskip.shape[2:]
            up = F.interpolate(x, size=tarSize, mode='trilinear', align_corners=False)
            cat = torch.cat([xskip, up], dim=1)
            x = self.IIM(self.Feat_extractor_2D(cat), self.Feat_extractor_3D(cat))
        else:
            x = self.IIM(self.Feat_extractor_2D(x), self.Feat_extractor_3D(x))
        return x

class AMFNet(nn.Module):
    def __init__(self, in_channel=1, n_classes=2):
        """
        Main architecture of the AMFNet model for feature extraction and segmentation.

        :param in_channel: Number of input channels (e.g., 1 for grayscale images)
        :param n_classes: Number of output classes for segmentation
        """
        super(AMFNet, self).__init__()

        # Define downsampling layers for encoder
        self.ec_layer1 = Layer_Down(self.in_channel, 64, downsample=False)
        self.ec_layer2 = Layer_Down(64, 128)
        self.ec_layer3 = Layer_Down(128, 256)
        self.ec_layer4 = Layer_Down(256, 512)

        # Define upsampling layers for decoder
        self.dc_layer4 = Layer_Up(256 + 512, 256)
        self.dc_layer3 = Layer_Up(128 + 256, 128)
        self.dc_layer2 = Layer_Up(64 + 128, 64)
        self.dc_layer1 = Layer_Up(64, 32, SKIP=False)
        self.dc_layer0 = nn.Conv3d(32, n_classes, kernel_size=(1, 1, 1), stride=1, padding=0, bias=False)

        # Additional pooling layer
        self.pool0 = nn.MaxPool3d(2)

    def forward(self, x):
        # Encoder pass
        feat_i = self.ec_layer1(x)
        feat_1 = self.pool0(feat_i)
        feat_2 = self.ec_layer2(feat_1)
        feat_3 = self.ec_layer3(feat_2)
        feat_4 = self.ec_layer4(feat_3)

        # Decoder pass
        dfeat_4 = self.dc_layer4([feat_4, feat_3])
        dfeat_3 = self.dc_layer3([dfeat_4, feat_2])
        dfeat_2 = self.dc_layer2([dfeat_3, feat_i])
        dfeat_1 = self.dc_layer1(dfeat_2)
        output = self.dc_layer0(dfeat_1)

        return output


class DoubleConv(nn.Module):
    def __init__(self, in_channels=3, out_channels=16, batch_normal=False):
        """
        Double convolutional layer, potentially with batch normalization.

        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param batch_normal: Whether to apply batch normalization
        """
        super(DoubleConv, self).__init__()
        channels = out_channels // 2
        if in_channels > out_channels:
            channels = in_channels // 2

        # Define layers in the double convolution
        layers = [
            nn.Conv3d(in_channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv3d(channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True)
        ]

        # Insert batch normalization if required
        if batch_normal:
            layers.insert(1, nn.BatchNorm3d(channels))
            layers.insert(len(layers) - 1, nn.BatchNorm3d(out_channels))

        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


class DownSampling(nn.Module):
    def __init__(self, in_channels=1, out_channels=16, batch_normal=False):
        """
        Downsampling module with max pooling followed by double convolution.

        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param batch_normal: Whether to apply batch normalization
        """
        super(DownSampling, self).__init__()
        self.maxpool_to_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels, batch_normal)
        )

    def forward(self, x):
        return self.maxpool_to_conv(x)


class UpSampling(nn.Module):
    def __init__(self, in_channels=1, out_channels=16, batch_normal=False, bilinear=False):
        """
        Upsampling module with bilinear interpolation or transposed convolution.

        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param batch_normal: Whether to apply batch normalization
        :param bilinear: Whether to use bilinear interpolation for upsampling
        """
        super(UpSampling, self).__init__()
        if bilinear:
            # Bilinear upsampling
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            # Transposed convolution for upsampling
            self.up = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=2, stride=2)

        # Apply double convolution after upsampling
        self.conv = DoubleConv(in_channels + in_channels // 2, out_channels, batch_normal)

    def forward(self, inputs1, inputs2):
        # Perform upsampling
        inputs1 = self.up(inputs1)
        # Concatenate with skip connection and apply convolution
        outputs = torch.cat([inputs1, inputs2], dim=1)
        outputs = self.conv(outputs)
        return outputs


class LastConv(nn.Module):
    def __init__(self, in_channels=1, out_channels=16):
        """
        Final convolutional layer to produce output.

        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        """
        super(LastConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, prev_nf=1, batch_normal=True, bilinear=False):
        """
        UNet architecture for 3D segmentation.

        :param in_channels: Number of input channels
        :param prev_nf: Number of output channels in the final layer
        :param batch_normal: Whether to apply batch normalization
        :param bilinear: Whether to use bilinear upsampling
        """
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.batch_normal = batch_normal
        self.bilinear = bilinear

        # Define the input convolutional layer and downsampling layers
        self.inputs = DoubleConv(in_channels, 64, self.batch_normal)
        self.down_1 = DownSampling(64, 128, self.batch_normal)
        self.down_2 = DownSampling(128, 256, self.batch_normal)
        self.down_3 = DownSampling(256, 512, self.batch_normal)

        # Define upsampling layers and final convolution
        self.up_1 = UpSampling(512, 256, self.batch_normal, self.bilinear)
        self.up_2 = UpSampling(256, 128, self.batch_normal, self.bilinear)
        self.up_3 = UpSampling(128, 64, self.batch_normal, self.bilinear)
        self.outputs = LastConv(64, prev_nf)

        self.final_nf = prev_nf

    def forward(self, x):
        # Encoder
        x1 = self.inputs(x)
        x2 = self.down_1(x1)
        x3 = self.down_2(x2)
        x4 = self.down_3(x3)

        # Decoder
        x5 = self.up_1(x4, x3)
        x6 = self.up_2(x5, x2)
        x7 = self.up_3(x6, x1)
        output = self.outputs(x7)

        return output


class SpatialTransformer(nn.Module):
    """
    N-Dimensional Spatial Transformer for deformable registration.

    :param size: Spatial size of the input
    :param mode: Interpolation mode (default is bilinear)
    """
    def __init__(self, size, mode='bilinear'):
        super().__init__()
        self.mode = mode

        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # Apply transformation to source image based on the flow field
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # Normalize grid values to [-1, 1]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # Reshape and permute grid dimensions for sampling
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        # Sample the source image at new locations
        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class Discriminator(nn.Module):
    def __init__(self, input_shape=(128, 128, 128)):
        """
        Discriminator network for GAN-based approaches.

        :param input_shape: Shape of the input volume
        """
        super(Discriminator, self).__init__()

        # Define the discriminator blocks with Conv3D, LeakyReLU, and Dropout
        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv3d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm3d(out_filters, 0.8))
            return block

        # Define the layers in the discriminator
        self.model = nn.Sequential(
            *discriminator_block(1, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # Calculate the downsampled size of the input
        ds_size_0 = input_shape[0] // (2 ** 4)
        ds_size_1 = input_shape[1] // (2 ** 4)
        ds_size_2 = input_shape[2] // (2 ** 4)

        # Define the final linear layer for outputting validity score
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size_0 * ds_size_1 * ds_size_2, 1))

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity


class Deformation_Net(nn.Module):
    def __init__(self, input_shape, in_channels=1, prev_nf=16, batch_normal=True, bilinear=False, bidir=False):
        """
        Deformation network for deformable image registration.

        :param input_shape: Shape of the input volume
        :param in_channels: Number of input channels
        :param prev_nf: Number of output channels in the final layer
        :param batch_normal: Whether to apply batch normalization
        :param bilinear: Whether to use bilinear upsampling
        :param bidir: Whether the network should be bidirectional
        """
        super(Deformation_Net, self).__init__()

        self.amf_net = AMFNet(in_channel=in_channels, n_classes=prev_nf)

        # Define the final convolution for flow field generation
        Conv = getattr(nn, 'Conv%dd' % 3)
        self.flow = Conv(16, 3, kernel_size=3, padding=1)

        # Initialize flow layer weights and biases
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # Define whether bidirectional training is used
        self.bidir = bidir

        # Define the transformer for spatial transformation
        self.transformer = SpatialTransformer(size=(input_shape[1], input_shape[2], input_shape[0]))

    def forward(self, source):
        """
        Forward pass through the deformation network.

        :param source: Source image tensor
        :return: Deformed source image and flow field
        """
        x = source
        x = self.amf_net(x)
        flow_field = self.flow(x)
        pos_flow = flow_field
        y_source = self.transformer(source, pos_flow)
        return y_source, pos_flow
