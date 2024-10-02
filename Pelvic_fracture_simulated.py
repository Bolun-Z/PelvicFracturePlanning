# ================================================================================
# Author: Bolun Zeng
# Input healthy pelvis images
# Output three results:
# 1. Segment the pelvis label into pieces based on the watershed algorithm.
# 2. Randomly mask parts of the fragments according to a probabilistic hyperparameter.
# 3. Randomly rotate the fragments except for the largest one.
# Objective: a. Generate simulated fracture data. b. Generate data for a self-supervised fracture repair model.
# ================================================================================

import os
import math
import torch
import random
import numpy as np
import SimpleITK as sitk
import torch.nn.functional as F
from numpy import random as np_random
from tqdm import tqdm

def affine_trans(img_array, theta_x, theta_y, theta_z, dx, dy, dz):
    """
    Affine transformation: rotate along x, y, z axes and translate by dx, dy, dz
    :param img: Image array
    :param theta_x: Rotation along x-axis
    :param theta_y: Rotation along y-axis
    :param theta_z: Rotation along z-axis
    :param dx: Translation along x-axis
    :param dy: Translation along y-axis
    :param dz: Translation along z-axis
    :return: Transformed image
    """
    img = torch.from_numpy(img_array.astype(np.float32)).unsqueeze(0).unsqueeze(0).float()
    rotate_x = torch.tensor([[1, 0, 0, 0],
                             [0, math.cos(theta_x), -math.sin(theta_x), 0],
                             [0, math.sin(theta_x), math.cos(theta_x), 0],
                             [0, 0, 0, 1]], dtype=torch.float)

    rotate_y = torch.tensor([[math.cos(theta_y), 0, math.sin(theta_y), 0],
                             [0, 1, 0, 0],
                             [-math.sin(theta_y), 0, math.cos(theta_y), 0],
                             [0, 0, 0, 1]], dtype=torch.float)

    rotate_z = torch.tensor([[math.cos(theta_z), -math.sin(theta_z), 0, 0],
                             [math.sin(theta_z), math.cos(theta_z), 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]], dtype=torch.float)

    trans = torch.tensor([[1, 0, 0, dx],
                          [0, 1, 0, dy],
                          [0, 0, 1, dz],
                          [0, 0, 0, 1]], dtype=torch.float)

    trans_matrix = torch.mm(trans, torch.mm(rotate_z, torch.mm(rotate_y, rotate_x)))
    trans_matrix_t = trans_matrix[:3, :]
    grid = F.affine_grid(trans_matrix_t.unsqueeze(0), img.size(), align_corners=False)
    output = F.grid_sample(img, grid, align_corners=False)

    output = output.squeeze().numpy()

    return output

def generate_BDAD_map(sitk_image, distance_map):
    """
    Calculate the BDAD map
    :param sitk_image: CT image in SimpleITK format
    :param distance_map: Distance map
    :return: BDAD map
    """
    # Random factor
    alpha = np_random.rand()  # Uniform distribution between 0-1
    alpha = alpha * 0.5 + 0.5

    # Calculate bone density probability map
    image_array = sitk.GetArrayFromImage(sitk_image)
    image_array_map = image_array.copy()
    image_array_map = image_array_map / np.max(image_array)
    image_array_map[image_array_map == 0] = 1
    image_array_map = image_array_map * (1 - alpha) + alpha

    # Bone density weighted distance map
    distance_map_array = sitk.GetArrayFromImage(distance_map)
    d_array = distance_map_array / image_array_map

    d_array[d_array > 0] = 0
    d_array_normalization = (d_array - np.min(d_array)) / (np.max(d_array) - np.min(d_array))
    d_array_normalization = d_array_normalization * (0 - np.min(distance_map_array)) + np.min(distance_map_array)

    BDAD_map = sitk.GetImageFromArray(d_array_normalization)

    return BDAD_map

def seg_by_watershed(sitk_image, sitk_label, level=2, mode='BDAD'):
    """
    Segment the bone fragments using the watershed algorithm
    :param sitk_image: CT image in SimpleITK format
    :param sitk_label: Label in SimpleITK format
    :param level: Watershed level
    :param mode: Map generation mode, either direct distance map or BDAD
    :return: Segmented watershed, classes, and number of classes
    """
    seg = sitk.ConnectedComponent(sitk_label > 0)  # Merge connected regions, segment unconnected regions
    filled = sitk.BinaryFillhole(seg != 0)  # Fill internal holes and apply distance mapping
    distance_map = sitk.SignedMaurerDistanceMap(filled, insideIsPositive=False, squaredDistance=False, useImageSpacing=True)

    distance_map_array = sitk.GetArrayFromImage(distance_map)
    distance_map_array[distance_map_array > 0] = 0
    distance_map = sitk.GetImageFromArray(distance_map_array)
    distance_map.CopyInformation(sitk_label)

    if mode == 'BDAD':
        sitk_map = generate_BDAD_map(sitk_image=sitk_image, distance_map=distance_map)
    else:
        sitk_map = distance_map

    sitk_watershed = sitk.MorphologicalWatershed(sitk_map, level=level, markWatershedLine=False, fullyConnected=True)

    sitk_watershed.CopyInformation(sitk_label)
    sitk_watershed = sitk.Mask(sitk_watershed, sitk.Cast(seg, sitk_watershed.GetPixelID()))
    watershed_array = sitk.GetArrayFromImage(sitk_watershed)
    classes = np.unique(watershed_array)
    num_classes = len(classes)

    return sitk_watershed, classes, num_classes

def cal_BBox(sitk_label):
    """
    :param sitk_label: SimpleITK label
    :return: Bounding box
    """
    label_array = sitk.GetArrayFromImage(sitk_label)
    label_array[label_array > 0] = 1
    label_t = sitk.GetImageFromArray(label_array)
    label_filter = sitk.LabelStatisticsImageFilter()
    label_filter.Execute(label_t, label_t)
    bbox_t = label_filter.GetBoundingBox(1)  # (xmin, xmax, ymin, ymax, zmin, zmax)

    bbox = (bbox_t[0], bbox_t[1], bbox_t[2], bbox_t[3], bbox_t[4], bbox_t[5])
    return bbox

def perform_affine_trans(sitk_image, sitk_label_frac, num_classes, theta_x, theta_y, theta_z, dx, dy, dz, expansion=30):
    """
    Affine transformation: rotate along x, y, z axes and translate by dx, dy, dz
    :param sitk_image: Original CT image -- Anatomical region ROI
    :param sitk_label_frac: Image after watershed segmentation -- Anatomical region ROI
    :param num_classes: Number of fragments
    :param theta_x: Rotation angle along x-axis
    :param theta_y: Rotation angle along y-axis
    :param theta_z: Rotation angle along z-axis
    :param dx: Translation along x-axis
    :param dy: Translation along y-axis
    :param dz: Translation along z-axis
    :param expansion: ROI expansion range
    :return: Transformed image and label
    """
    # The largest fragment will not be transformed
    lss_filter = sitk.LabelShapeStatisticsImageFilter()
    lss_filter.Execute(sitk_label_frac)  # Analyze connected regions
    volumes = []
    for j in range(1, num_classes):
        volume = lss_filter.GetNumberOfPixels(j)
        volumes.append(volume)
    class_max_volume = np.argmax(volumes)

    # Random rotation and transformation
    data_trans = list(np.arange(1, num_classes))
    data_trans.remove(class_max_volume + 1)

    frac_label_array = sitk.GetArrayFromImage(sitk_label_frac)
    input_image_array = sitk.GetArrayFromImage(sitk_image)

    # Create an empty array to store the transformed label
    output_frac_trans_array = np.zeros(
        shape=[frac_label_array.shape[0] + expansion, frac_label_array.shape[1] + expansion, frac_label_array.shape[2] + expansion])

    # Create an empty array to store the transformed image
    output_image_trans_array = np.zeros_like(output_frac_trans_array)

    # Data transformation
    for i in range(1, num_classes + 1):
        frac_label_array_t = frac_label_array.copy()
        frac_label_array_t[frac_label_array_t != i] = 0

        # Padding to prevent boundary overflow after rotation
        # Create label data for transformation
        output_frac_trans_array_t = np.zeros(
            shape=[frac_label_array.shape[0] + expansion, frac_label_array.shape[1] + expansion, frac_label_array.shape[2] + expansion])
        output_frac_trans_array_t[expansion//2:expansion//2 + frac_label_array.shape[0], expansion//2:expansion//2 + frac_label_array.shape[1],
        expansion // 2:expansion // 2 + frac_label_array.shape[2]] = frac_label_array_t

        # Create image data for transformation
        output_image_trans_array_t = np.zeros(
            shape=[frac_label_array.shape[0] + expansion, frac_label_array.shape[1] + expansion, frac_label_array.shape[2] + expansion])
        output_image_trans_array_t[expansion//2:expansion//2 + frac_label_array.shape[0], expansion//2:expansion//2 + frac_label_array.shape[1],
        expansion // 2:expansion // 2 + frac_label_array.shape[2]] = input_image_array

        output_image_trans_array_t[output_frac_trans_array_t == 0] = 0  # Extract the CT image of the current anatomical region

        if i in data_trans:
            output_frac_trans_array_t = affine_trans(output_frac_trans_array_t, theta_x, theta_y, theta_z, dx, dy, dz)
            output_image_trans_array_t = affine_trans(output_image_trans_array_t, theta_x, theta_y, theta_z, dx, dy, dz)

            output_frac_trans_array_t[output_frac_trans_array_t > 0.5] = i

        # Label after random transformation
        overlap_indices = np.nonzero(output_frac_trans_array_t * output_frac_trans_array)
        output_frac_trans_array = output_frac_trans_array + output_frac_trans_array_t
        output_frac_trans_array[overlap_indices] = class_max_volume + 1  # Set overlapping areas to the largest volume class

        # Image after random transformation
        output_image_trans_array = output_image_trans_array + output_image_trans_array_t
        pixel_max = np.max(input_image_array)
        output_image_trans_array[output_image_trans_array > pixel_max] = pixel_max  # Set overlapping pixels to the maximum pixel value

    output_image_trans = sitk.GetImageFromArray(output_image_trans_array)
    output_image_trans.SetOrigin(sitk_image.GetOrigin())
    output_image_trans.SetSpacing(sitk_image.GetSpacing())
    output_image_trans.SetDirection(sitk_image.GetDirection())

    output_frac_trans = sitk.GetImageFromArray(output_frac_trans_array)
    output_frac_trans.SetOrigin(sitk_image.GetOrigin())
    output_frac_trans.SetSpacing(sitk_image.GetSpacing())
    output_frac_trans.SetDirection(sitk_image.GetDirection())

    return output_image_trans, output_frac_trans

def reset_category(anatomy_class, sitk_frac):
    """
    Reset the category of the fragments after transformation.
    :param anatomy_class: Current anatomical region class
    :param sitk_frac: Fracture label in SimpleITK format
    :return: Fracture label with reset categories in SimpleITK format
    """
    sitk_frac = sitk.Cast(sitk_frac, sitk.sitkUInt8)
    frac_array = sitk.GetArrayFromImage(sitk_frac)

    fragment_list = np.unique(frac_array)
    fragment_list = fragment_list[fragment_list != 0]

    # Find the fragment with the largest volume
    lss_filter = sitk.LabelShapeStatisticsImageFilter()
    lss_filter.Execute(sitk_frac)  # Analyze connected regions
    volumes = []

    for j in range(1, len(fragment_list) + 1):
        volume = lss_filter.GetNumberOfPixels(j)
        volumes.append(volume)
    class_max_volume = fragment_list[np.argmax(volumes)]

    # Reset other small fragments' categories to anatomy_class * 10 + i + 1
    fragment_list = fragment_list[fragment_list != class_max_volume]
    for i, category in enumerate(fragment_list):
        # Find the indices of the elements belonging to the current category
        indices = np.where(frac_array == category)
        frac_array[indices] = anatomy_class * 10 + i + 1

    # Reset the largest fragment to anatomy_class
    frac_array[frac_array == class_max_volume] = anatomy_class

    output_sitk_frac = sitk.GetImageFromArray(frac_array)
    output_sitk_frac.CopyInformation(sitk_frac)

    return output_sitk_frac

def apply(image_path, label_path, theta_x, theta_y, theta_z, dx, dy, dz, target=[1, 2, 3], level=2, mode='BDAD', expansion=30):
    """
    Perform the fracture generation process
    :param image_path: Path to the CT image
    :param label_path: Path to the label
    :param theta_x: Rotation angle for x-axis
    :param theta_y: Rotation angle for y-axis
    :param theta_z: Rotation angle for z-axis
    :param dx: Translation for x-axis
    :param dy: Translation for y-axis
    :param dz: Translation for z-axis
    :param target: Target anatomical regions (1-Sacrum, 2-Left Ilium, 3-Right Ilium)
    :param level: Watershed level
    :param mode: Map generation mode (BDAD or distance map)
    :param expansion: ROI expansion range
    :return: Transformed output image and label
    """
    input_image = sitk.ReadImage(image_path, sitk.sitkFloat32)
    input_label = sitk.ReadImage(label_path, sitk.sitkUInt8)

    input_image_array = sitk.GetArrayFromImage(input_image)
    input_label_array = sitk.GetArrayFromImage(input_label)

    # Calculate the number of anatomical regions in the input pelvis
    anatomy_list = np.unique(sitk.GetArrayFromImage(input_label))
    anatomy_list = anatomy_list[anatomy_list != 0]

    target_anatomy_list = list(set(anatomy_list) & set(target))  # Only calculate the target anatomical regions
    keep_anatomy_list = list(set(anatomy_list) - set(target_anatomy_list))  # Keep the unchanged anatomical regions

    output_image_array = np.zeros_like(input_image_array)
    output_label_array = np.zeros_like(input_label_array)

    for i, cla in enumerate(tqdm(keep_anatomy_list)):
        input_image_array_i = sitk.GetArrayFromImage(input_image)
        input_label_array_i = sitk.GetArrayFromImage(input_label)

        input_label_array_i[input_label_array_i != cla] = 0
        input_image_array_i[input_label_array_i == 0] = 0

        output_image_array = output_image_array + input_image_array_i
        output_label_array = output_label_array + input_label_array_i

    for i, cla in enumerate(tqdm(target_anatomy_list)):
        # Calculate the ROI for the current anatomical region, and extract the corresponding image and label
        input_label_i_array = sitk.GetArrayFromImage(input_label)
        input_label_i_array[input_label_i_array != cla] = 0
        input_label_i = sitk.GetImageFromArray(input_label_i_array)
        bbox = cal_BBox(input_label_i)

        input_label_i_roi_array = input_label_i_array[bbox[4]: bbox[5], bbox[2]:bbox[3], bbox[0]:bbox[1]]
        input_label_i_roi = sitk.GetImageFromArray(input_label_i_roi_array)
        input_label_i_roi.SetSpacing(input_image.GetSpacing())
        input_label_i_roi.SetDirection(input_image.GetDirection())
        input_label_i_roi.SetOrigin(input_image.GetOrigin())

        input_image_i_array = sitk.GetArrayFromImage(input_image)
        input_image_i_roi_array = input_image_i_array[bbox[4]: bbox[5], bbox[2]:bbox[3], bbox[0]:bbox[1]]
        input_image_i_roi = sitk.GetImageFromArray(input_image_i_roi_array)
        input_image_i_roi.SetSpacing(input_image.GetSpacing())
        input_image_i_roi.SetDirection(input_image.GetDirection())
        input_image_i_roi.SetOrigin(input_image.GetOrigin())

        # Segment the bone fragments
        frac_division, classes, num_classes = seg_by_watershed(sitk_image=input_image_i_roi, sitk_label=input_label_i_roi, level=level, mode=mode)

        # Perform transformation on the current anatomical region
        output_image_trans, output_frac_trans = perform_affine_trans(sitk_image=input_image_i_roi, sitk_label_frac=frac_division,
                                                                     num_classes=num_classes, theta_x=theta_x, theta_y=theta_y,
                                                                     theta_z=theta_z, dx=dx, dy=dy, dz=dz, expansion=expansion)

        output_frac_trans = reset_category(anatomy_class=cla, sitk_frac=output_frac_trans)

        # Store the transformed results of the current anatomical region
        output_image_array_t = np.zeros_like(input_image_array)
        output_label_array_t = np.zeros_like(input_label_array)

        output_points = np.zeros_like(bbox)  # Final restoration coordinates
        bias = np.zeros_like(bbox)  # Calculate offset if transformation exceeds the original image boundary
        output_points[0] = max(bbox[0] - expansion // 2, 0)
        bias[0] = expansion // 2 - np.abs(output_points[0] - bbox[0])
        output_points[1] = min(bbox[1] + expansion // 2, output_image_array.shape[2])
        bias[1] = expansion // 2 - np.abs(output_points[1] - bbox[1])

        output_points[2] = max(bbox[2] - expansion // 2, 0)
        bias[2] = expansion // 2 - np.abs(output_points[2] - bbox[2])
        output_points[3] = min(bbox[3] + expansion // 2, output_image_array.shape[1])
        bias[3] = expansion // 2 - np.abs(output_points[3] - bbox[3])

        output_points[4] = max(bbox[4] - expansion // 2, 0)
        bias[4] = expansion // 2 - np.abs(output_points[4] - bbox[4])
        output_points[5] = min(bbox[5] + expansion // 2, output_image_array.shape[0])
        bias[5] = expansion // 2 - np.abs(output_points[5] - bbox[5])

        x, y, z = sitk.GetArrayFromImage(output_image_trans).shape

        output_image_array_t[output_points[4]: output_points[5], output_points[2]: output_points[3], output_points[0]: output_points[1]] \
            = sitk.GetArrayFromImage(output_image_trans)[bias[4]: x - bias[5], bias[2]: y - bias[3], bias[0]: z - bias[1]]

        output_label_array_t[output_points[4]: output_points[5], output_points[2]:output_points[3], output_points[0]: output_points[1]] \
            = sitk.GetArrayFromImage(output_frac_trans)[bias[4]: x - bias[5], bias[2]: y - bias[3], bias[0]: z - bias[1]]

        # Add to final output
        overlap_indices = np.nonzero(output_label_array * output_label_array_t)
        output_label_array = output_label_array + output_label_array_t
        output_label_array[overlap_indices] = cla  # Set the overlapping region to the current category

        output_image_array = output_image_array + output_image_array_t
        pixel_max = np.max(input_image_array)
        output_image_array[output_image_array > pixel_max] = pixel_max  # Set the overlapping region pixel to the maximum pixel value

    output_image = sitk.GetImageFromArray(output_image_array)
    output_image.CopyInformation(input_image)
    output_label = sitk.GetImageFromArray(output_label_array)
    output_label.CopyInformation(input_label)

    return output_image, output_label

if __name__ == '__main__':
    # Example case
    image_path = ''
    label_path = ''

    theta_x = random.uniform(-math.pi / 12, math.pi / 12)
    theta_y = random.uniform(-math.pi / 12, math.pi / 12)
    theta_z = random.uniform(-math.pi / 12, math.pi / 12)
    dx = random.uniform(-0.0005, 0.0005)
    dy = random.uniform(-0.0005, 0.0005)
    dz = random.uniform(-0.0005, 0.0005)

    output_image, output_label = apply(image_path, label_path, theta_x, theta_y, theta_z, dx, dy, dz, target=[1, 3],
                                       level=3, mode='BDAD', expansion=30)

    sitk.WriteImage(output_image, 'output_image.nii.gz')
    sitk.WriteImage(output_label, 'output_label.nii.gz')
