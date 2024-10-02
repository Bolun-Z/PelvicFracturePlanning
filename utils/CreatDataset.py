from torch.utils.data.dataset import Dataset
import SimpleITK as sitk
import os
import numpy as np
import torch
import random
from skimage.metrics import structural_similarity as ssim

class CreatDataset(Dataset):
    def __init__(self, dataset_lines, input_shape, mode='Train'):
        """
        Initialize the dataset class for loading medical images.

        :param dataset_lines: List of file paths for the dataset
        :param input_shape: Shape of the input images (depth, height, width)
        :param mode: Mode of the dataset (default is 'Train')
        """
        super(CreatDataset, self).__init__()
        self.dataset_lines = dataset_lines
        self.input_shape = input_shape
        self.mode = mode

    def __getitem__(self, index):
        """
        Load one pair of healthy and simulated fractured images for training.

        :param index: Index of the file path in dataset_lines
        :return: Tuple containing the simulated and healthy images as numpy arrays
        """
        file_path = self.dataset_lines[index]

        """
        Read the healthy template image.
        """
        healthy_image_path = file_path
        healthy_image = sitk.ReadImage(healthy_image_path, sitk.sitkFloat32)
        healthy_image = self.resize_image(healthy_image, [self.input_shape[1], self.input_shape[2], self.input_shape[0]], is_label=False)  # Resampling
        healthy_image_array = sitk.GetArrayFromImage(healthy_image)
        healthy_image_array = np.expand_dims(np.array(healthy_image_array), 0)  # Add a channel dimension

        """
        Read the simulated fractured data.
        """
        simulation_image_path = file_path[:37] + 'random_trans' + file_path[42:]  # Modify the file path for simulated data
        simulation_image = sitk.ReadImage(simulation_image_path, sitk.sitkFloat32)
        simulation_image = self.resize_image(simulation_image, [self.input_shape[1], self.input_shape[2], self.input_shape[0]], is_label=False)  # Resampling
        simulation_image_array = sitk.GetArrayFromImage(simulation_image)
        simulation_image_array = np.expand_dims(np.array(simulation_image_array), 0)  # Add a channel dimension

        return simulation_image_array, healthy_image_array

    def __len__(self):
        """
        Return the total number of samples in the dataset.

        :return: Length of dataset_lines
        """
        return len(self.dataset_lines)

    def load_data(self):
        """
        A method to load the data (used when initializing DataLoader).

        :return: self (dataset object)
        """
        return self

    def resize_image(self, sitk_image, new_size=[256, 512, 1], is_label=False):
        """
        Resample the image to the desired new size.

        :param sitk_image: Input SimpleITK image object
        :param new_size: Desired size for the output image (depth, height, width)
        :param is_label: Whether the image is a label (for choosing the interpolation method)
        :return: Resampled image
        """
        size = np.array(sitk_image.GetSize())
        spacing = np.array(sitk_image.GetSpacing())
        new_size = np.array(new_size)
        new_spacing_refine = size * spacing / new_size
        new_spacing_refine = [float(s) for s in new_spacing_refine]
        new_size = [int(s) for s in new_size]

        resample = sitk.ResampleImageFilter()
        resample.SetOutputDirection(sitk_image.GetDirection())
        resample.SetOutputOrigin(sitk_image.GetOrigin())
        resample.SetSize(new_size)
        resample.SetOutputSpacing(new_spacing_refine)

        # Choose interpolation based on whether the image is a label or not
        if is_label:
            resample.SetInterpolator(sitk.sitkNearestNeighbor)  # For labels, use nearest neighbor interpolation
        else:
            resample.SetInterpolator(sitk.sitkLinear)  # For images, use linear interpolation

        itk_image = resample.Execute(sitk_image)
        return itk_image


# Custom collate function for DataLoader (used to combine samples in batches)
def dataset_collate(batch):
    """
    Collate function for custom dataset. Combines multiple batches and prepares them for the DataLoader.

    :param batch: List of batches (each containing simulated image, healthy image, and labels)
    :return: Tuple of tensors containing batched images and labels
    """
    left_images = []
    right_images = []
    labels_similarity = []
    labels_type_x1 = []
    labels_type_x2 = []

    # Loop through each batch and append data to corresponding lists
    for pair_imgs, pair_labels_similarity, pair_labels_type_x1, pair_labels_type_x2 in batch:
        left_images.append(pair_imgs[0])
        right_images.append(pair_imgs[1])
        labels_similarity.append(pair_labels_similarity)
        labels_type_x1.append(pair_labels_type_x1)
        labels_type_x2.append(pair_labels_type_x2)

    # Convert lists to tensors
    images = torch.from_numpy(np.array([left_images, right_images])).type(torch.FloatTensor)
    labels_similarity = torch.from_numpy(np.array(labels_similarity)).type(torch.FloatTensor)
    labels_type_x1 = torch.from_numpy(np.array(labels_type_x1)).type(torch.FloatTensor)
    labels_type_x2 = torch.from_numpy(np.array(labels_type_x2)).type(torch.FloatTensor)

    return images, labels_similarity, labels_type_x1, labels_type_x2
