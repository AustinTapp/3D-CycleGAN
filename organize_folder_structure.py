import os
import shutil
from time import time
import re
import argparse
import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndimage
from utils.NiftiDataset import *


def ct_window(image, center=50, width=100):
    low_bound = center - width // 2
    up_bound = center + width // 2
    if str(image.dtype).split(".")[0] == "torch":
        windowed_image = image.clone()
    else:
        windowed_image = image.copy()
    windowed_image[windowed_image < low_bound] = low_bound
    windowed_image[windowed_image > up_bound] = up_bound
    windowed_image -= windowed_image.min()
    windowed_image /= (windowed_image.max() - windowed_image.min())
    return windowed_image

def Normalization(image):
    """
    Normalize an image to 0 - 255 (8bits)
    """
    normalizeFilter = sitk.NormalizeImageFilter()
    resacleFilter = sitk.RescaleIntensityImageFilter()
    resacleFilter.SetOutputMaximum(255)
    resacleFilter.SetOutputMinimum(0)

    image = normalizeFilter.Execute(image)  # set mean and std deviation
    image = resacleFilter.Execute(image)  # set intensity 0-255

    return image

def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def lstFiles(Path):
    images_list = []  # create an empty list, the raw image data files is stored here
    for dirName, subdirList, fileList in os.walk(Path):
        for filename in fileList:
            if ".nii.gz" in filename.lower():
                images_list.append(os.path.join(dirName, filename))
            elif ".nii" in filename.lower():
                images_list.append(os.path.join(dirName, filename))
            elif ".mhd" in filename.lower():
                images_list.append(os.path.join(dirName, filename))

    images_list = sorted(images_list, key=numericalSort)
    return images_list


def Align(image, reference):
    image_array = sitk.GetArrayFromImage(image)

    label_origin = reference.GetOrigin()
    label_direction = reference.GetDirection()
    label_spacing = reference.GetSpacing()

    image = sitk.GetImageFromArray(image_array)
    image.SetOrigin(label_origin)
    image.SetSpacing(label_spacing)
    image.SetDirection(label_direction)

    return image


def CropBackground(image, label):
    size_new = (240, 240, 120)

    def Normalization(image):
        """
        Normalize an image to 0 - 255 (8bits)
        """
        normalizeFilter = sitk.NormalizeImageFilter()
        resacleFilter = sitk.RescaleIntensityImageFilter()
        resacleFilter.SetOutputMaximum(255)
        resacleFilter.SetOutputMinimum(0)

        image = normalizeFilter.Execute(image)  # set mean and std deviation
        image = resacleFilter.Execute(image)  # set intensity 0-255

        return image

    image2 = Normalization(image)
    label2 = Normalization(label)

    threshold = sitk.BinaryThresholdImageFilter()
    threshold.SetLowerThreshold(20)
    threshold.SetUpperThreshold(255)
    threshold.SetInsideValue(1)
    threshold.SetOutsideValue(0)

    roiFilter = sitk.RegionOfInterestImageFilter()
    roiFilter.SetSize([size_new[0], size_new[1], size_new[2]])

    image_mask = threshold.Execute(image2)
    image_mask = sitk.GetArrayFromImage(image_mask)
    image_mask = np.transpose(image_mask, (2, 1, 0))

    import scipy
    centroid = scipy.ndimage.measurements.center_of_mass(image_mask)

    x_centroid = np.int(centroid[0])
    y_centroid = np.int(centroid[1])

    roiFilter.SetIndex([int(x_centroid - (size_new[0]) / 2), int(y_centroid - (size_new[1]) / 2), 0])

    label_crop = roiFilter.Execute(label)
    image_crop = roiFilter.Execute(image)

    return image_crop, label_crop


def Registration(image, label, ct):
    image, image_sobel, label, label_sobel, = image, image, label, label

    Gaus = sitk.GradientMagnitudeRecursiveGaussianImageFilter()
    image_sobel = Gaus.Execute(image_sobel)
    label_sobel = Gaus.Execute(label_sobel)

    fixed_image = label_sobel
    moving_image = image_sobel

    initial_transform = sitk.CenteredTransformInitializer(fixed_image,
                                                          moving_image,
                                                          sitk.Euler3DTransform(),
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)

    registration_method = sitk.ImageRegistrationMethod()
    # registration_method.SetMetricAs
    # Similarity metric settings.
    # registration_method.SetMetricAsJointHistogramMutualInformation(numberOfHistogramBins=60, varianceForJointPDFSmoothing=0.25)
    if (ct):
        registration_method.SetMetricAsANTSNeighborhoodCorrelation(2)
    else:
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50) # original, was 50

    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.1)
    registration_method.SetInterpolator(sitk.sitkLinear)
    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100,
                                                      convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32),
                                                  sitk.Cast(moving_image, sitk.sitkFloat32))

    image = sitk.Resample(image, fixed_image, final_transform, sitk.sitkLinear, 0.0,
                          moving_image.GetPixelID())

    return image, label


parser = argparse.ArgumentParser()
parser.add_argument('--images', default='./Data_folder/CTnoBed', help='path to the CTs')
parser.add_argument('--labels', default='./Data_folder/T1', help='path to the T1s')
parser.add_argument('--split', default=0, help='number of images for testing')
parser.add_argument('--resolution', default=(1, 1, 1), help='new resolution to resample the all data')
args = parser.parse_args()

if __name__ == "__main__":

    list_images = lstFiles(args.images)
    list_labels = lstFiles(args.labels)

    #reference_image = list_images[2]  # setting a reference image to have all data in the same coordinate system, was list_labels[0]
    reference_image = sitk.ReadImage("./Data_folder/CTnoBed/76b_noBed.nii.gz")

    print(reference_image)
    reference_image = resample_sitk_image(reference_image, spacing=args.resolution, interpolator='linear')

    if not os.path.isdir('./Data_folder/train'):
        os.mkdir('./Data_folder/train')

    if not os.path.isdir('./Data_folder/test'):
        os.mkdir('./Data_folder/test')

    for i in range(len(list_images) - int(args.split)):

        save_directory_images = './Data_folder/train/images'
        #save_directory_labels = "C:/Users/pmilab/Desktop/preprocessed ctmri paired/mri"
        #save_directory_images = "C:/Users/pmilab/Desktop/preprocessed ctmri paired/ct"
        save_directory_labels = './Data_folder/train/labels'

        if not os.path.isdir(save_directory_images):
            os.mkdir(save_directory_images)

        if not os.path.isdir(save_directory_labels):
            os.mkdir(save_directory_labels)

        """if int(args.split)+i >= len(list_labels):
            a = list_images[int(args.split) + i]
            image = sitk.ReadImage(a)
        else:
            b = list_labels[int(args.split) + i]
            a = list_images[int(args.split) + i]
            label = sitk.ReadImage(b)
            image = sitk.ReadImage(a)"""
        b = list_labels[int(args.split) + i]
        a = list_images[int(args.split) + i]

        print(a)

        label = sitk.ReadImage(b)
        image = sitk.ReadImage(a)

        image = resample_sitk_image(image, spacing=args.resolution, interpolator='linear')
        label = resample_sitk_image(label, spacing=args.resolution, interpolator='linear')

        label, reference_image = Registration(label, reference_image, False)
        image, reference_image = Registration(image, reference_image, True)  # new
        # image, label = Registration(image, label)

        """image = resample_sitk_image(image, spacing=args.resolution, interpolator='linear')
        label = resample_sitk_image(label, spacing=args.resolution, interpolator='linear')"""

        image = Align(image, reference_image)
        label = Align(label, reference_image)

        """# normalize
                image = Normalization(image)
                label = Normalization(label)"""

        image = sitk.GetArrayFromImage(image)
        image = ct_window(image)

        """label = sitk.GetArrayFromImage(label)
        label /= (label.max()-label.min())"""

        """# masking
        segmentation = sitk.BinaryThresholdImageFilter()
        segmentation.SetLowerThreshold(100) # tbd
        mask = segmentation.Execute(image)"""
        # label = sitk.GetImageFromArray(label)

        image = sitk.GetImageFromArray(image)

        label_directory = os.path.join(str(save_directory_labels), str(i) + '.nii')
        image_directory = os.path.join(str(save_directory_images), str(i) + '.nii')

        """if int(args.split)+i >= len(list_labels):
            sitk.WriteImage(image, image_directory)
        else:
            sitk.WriteImage(image, image_directory)
            sitk.WriteImage(label, label_directory)"""

        sitk.WriteImage(image, image_directory)
        sitk.WriteImage(label, label_directory)

    for i in range(int(args.split)):

        save_directory_images = './Data_folder/test/images'
        save_directory_labels = './Data_folder/test/labels'

        if not os.path.isdir(save_directory_images):
            os.mkdir(save_directory_images)

        if not os.path.isdir(save_directory_labels):
            os.mkdir(save_directory_labels)

        a = list_images[i]
        b = list_labels[i]

        print(a)

        label = sitk.ReadImage(b)
        image = sitk.ReadImage(a)

        image = resample_sitk_image(image, spacing=args.resolution, interpolator='linear')
        label = resample_sitk_image(label, spacing=args.resolution, interpolator='linear')

        label, reference_image = Registration(label, reference_image, False)
        image, reference_image = Registration(image, reference_image, True)  # new
        # image, label = Registration(image, label)

        """image = resample_sitk_image(image, spacing=args.resolution, interpolator='linear') 
        label = resample_sitk_image(label, spacing=args.resolution, interpolator='linear') """

        image = Align(image, reference_image)
        label = Align(label, reference_image)

        image = sitk.GetArrayFromImage(image)
        image = ct_window(image)
        image = sitk.GetImageFromArray(image)

        label_directory = os.path.join(str(save_directory_labels), str(i) + '.nii.gz')
        image_directory = os.path.join(str(save_directory_images), str(i) + '.nii.gz')

        sitk.WriteImage(image, image_directory)
        sitk.WriteImage(label, label_directory)
