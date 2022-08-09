import SimpleITK as sitk
import matplotlib.pyplot as plt
import torch
from torchmetrics import MeanAbsoluteError
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics import StructuralSimilarityIndexMeasure
import gc

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
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)  # original, was 50

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


if __name__ == '__main__':
    # loading images
    reader = sitk.ImageFileReader()
    reader.SetFileName(
        "C:/Users/pmilab/Desktop/preprocessed ctmri paired/ct/0.nii")  # ground truth
    true = reader.Execute()
    true = Normalization(true)

    reader.SetFileName(
        "C:/Users/pmilab/Desktop/preprocessed ctmri paired/ct/segment-component2-mse.nii")
    # reader.SetFileName("C:/Users/pmilab/PycharmProjects/3D-CycleGan-Pytorch-MedImaging-main/Data_folder/test/labels/0.nii") # output result
    result = reader.Execute()
    result = Normalization(result)
    result, true = Registration(true, result, True)

    result = sitk.GetArrayFromImage(result)
    print("result shape", result.shape)

    true = sitk.GetArrayFromImage(true)
    print("true shape", true.shape)
    # print(true[0])
    print("true slice", true[0].shape, "\nresult slice", result[0].shape)

    for i in range(result[:, 0, 0].size):
        for j in range(result[0, :, 0].size):
            for k in range(result[0, 0, :].size):
                if result[i, j, k] < 0:
                    result[i, j, k] = 0
                if true[i, j, k] < 0:
                    true[i, j, k] = 0

    # histogram
    histA = []  # ground truth
    histB = []  # result

    for i in range(true[:, 0, 0].size):
        sliceA = true[i]
        sliceB = result[i]
        valA = plt.hist(sliceA.ravel(), bins=range(256))
        valB = plt.hist(sliceB.ravel(), bins=range(256))
        histA.append(valA[0])
        histB.append(valB[0])

    # print(len(histB[0]))
    # print(len(histA[0]))


    print("histA", histA[0])
    print("histB", histB[0])

    histC = []  # array of differences of histograms
    for i in range(len(histA)):
        histC.append(histB[i] - histA[i])

    histDiff = []  # total difference of histograms
    for i in range(len(histC[0])):
        value = 0
        for j in range(len(histC)):
            value += histC[j][i]
        histDiff.append(value)
    print(histDiff)

    print("histDiff length", len(histDiff))
    plt.clf()

    plt.plot(histDiff[1:255])

    # print(histC)
    plt.show()

    # mean absolute error
    result_tensor = torch.from_numpy(result)
    result_tensor = torch.tensor(result_tensor, dtype=torch.float64)
    print()
    true_tensor = torch.from_numpy(true)
    mae = MeanAbsoluteError()
    mae = mae(result_tensor.cpu(), true_tensor.cpu())
    print("mae from torch", mae.numpy())

    MAE = sum(sum(sum(abs(result - true)))) / result.size
    print("mae from numpy", MAE)


    psnr = PeakSignalNoiseRatio(reduction=None)
    psnr = psnr(result_tensor.cpu(), true_tensor.cpu())
    print("psnr", psnr.numpy())

    ssim = StructuralSimilarityIndexMeasure()
    ssim = ssim(result_tensor.unsqueeze(0).cpu(), true_tensor.unsqueeze(0).cpu())
    print("ssim", ssim.numpy())

    fid = FrechetInceptionDistance()  # unsure, this seems like it wants all the images possible
    result_tensor.unsqueeze_(1)
    result_tensor = result_tensor.repeat(1, 3, 1, 1)
    true_tensor.unsqueeze_(1)
    true_tensor = true_tensor.repeat(1, 3, 1, 1)
    fid.update(result_tensor.type(torch.uint8), real=False)
    fid.update(true_tensor.type(torch.uint8), real=True)
    fid = fid.compute()
    print("fid", fid.numpy())
