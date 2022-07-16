/*=========================================================================
 *
 *  Copyright Austin Tapp - SZI Children's National
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         https://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#include <cstdlib>
#include <string>


#include "itkCastImageFilter.h"
#include "itkCheckerBoardImageFilter.h"
#include "itkDiscreteGaussianImageFilter.h"
#include "itkEllipseSpatialObject.h"
#include "itkExtractImageFilter.h"
#include "itkImage.h"
#include "itkImageToHistogramFilter.h"
#include "itkImageFileReader.h"
#include "itkImageRegionIterator.h"
#include "itkImageRegistrationMethod.h"
#include "itkResampleImageFilter.h"
#include "itkGradientDescentOptimizer.h"
#include "itkMutualInformationImageToImageMetric.h"
#include "itkNiftiImageIO.h"
#include "itkNormalizeImageFilter.h"
#include "itkNumericSeriesFileNames.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkSpatialObjectToImageFilter.h"
#include "itkSubtractImageFilter.h"
#include "itkTranslationTransform.h"



itk::ImageIOBase::Pointer getImageIO(std::string input) {
    itk::ImageIOBase::Pointer imageIO = itk::ImageIOFactory::CreateImageIO(input.c_str(), itk::ImageIOFactory::ReadMode);

    imageIO->SetFileName(input);
    imageIO->ReadImageInformation();

    return imageIO;
}

itk::ImageIOBase::IOComponentType component_type(itk::ImageIOBase::Pointer imageIO) {
    return imageIO->GetComponentType();
}

itk::ImageIOBase::IOPixelType pixel_type(itk::ImageIOBase::Pointer imageIO) {
    return imageIO->GetPixelType();
}

size_t num_dimensions(itk::ImageIOBase::Pointer imageIO) {
    return imageIO->GetNumberOfDimensions();
}

int
main(int argc, char* argv[])
{
    if (argc != 4)
    {
        std::cerr << "Usage: " << std::endl;
        std::cerr << argv[0];
        std::cerr << " <ground_truth> <predicted_image> <number_of_bins>";
        std::cerr << std::endl;
        return EXIT_FAILURE;
    }

    //const char* ground_truth = argv[1];
    //const char* predicted_image = argv[2];

    //itk::NiftiImageIO::Pointer GTimage = getImageIO(ground_truth);
    //itk::NiftiImageIO::Pointer Pdimage = getImageIO(predicted_image);

    constexpr unsigned int Dimension = 3;

    using PixelType = unsigned char;
    using GTImageType = itk::Image<PixelType, Dimension>;
    using PdImageType = itk::Image<PixelType, Dimension>;

    GTImageType::Pointer GTinput = itk::ReadImage<GTImageType>(argv[1]);
    PdImageType::Pointer Pdinput = itk::ReadImage<PdImageType>(argv[2]);


    // In order to read a image, we need its dimensionality and component type
    //std::cout << "numDimensions: " << num_dimensions(imageIO) << std::endl;
    //std::cout << "component type: " << imageIO->GetComponentTypeAsString(component_type(imageIO)) << std::endl;
    //std::cout << "pixel type: " << imageIO->GetPixelTypeAsString(pixel_type(imageIO)) << std::endl;

    //read the GT image
    //constexpr unsigned int GTDimension = num_dimensions(GTimage);
    //constexpr unsigned int GTCompType = component_type(GTimage);
    //constexpr unsigned char GTPixelType = pixel_type(GTimage);
    //using GTImageType = itk::Image<GTPixelType, GTDimension>;
    //const auto GTinput = itk::Read<GTImageType>(ground_truth);

    //read the predicted image
    //constexpr unsigned int PdDimension = num_dimensions(Pdimage);
    //constexpr unsigned int PdCompType = component_type(Pdimage);
    //constexpr unsigned char PdPixelType = pixel_type(Pdimage);
    //using PdImageType = itk::Image<PdPixelType, PdDimension>;
    //const auto PdInput = itk::Read<PdImageType>(predicted_image);

    //rescale the GT image (pixel intensity)

    using GTRescaleImageType = itk::Image<PixelType, Dimension>;
    using RescaleFilterType = itk::RescaleIntensityImageFilter<GTImageType, GTRescaleImageType>;
    auto GTrescale = RescaleFilterType::New();
    GTrescale->SetInput(GTinput);
    GTrescale->SetOutputMinimum(0);
    GTrescale->SetOutputMaximum(255);
    GTrescale->UpdateLargestPossibleRegion();

    //rescale the predicted image (pixel intensity)
    using PdRescaleFilterType = itk::RescaleIntensityImageFilter<GTImageType, GTRescaleImageType>;
    auto Pdrescale = PdRescaleFilterType::New();
    Pdrescale->SetInput(Pdinput);
    Pdrescale->SetOutputMinimum(0);
    Pdrescale->SetOutputMaximum(255);
    Pdrescale->UpdateLargestPossibleRegion();

    //prepare extraction filter for GT and predicted
    using ExtractFilterType = itk::ExtractImageFilter<GTImageType, GTImageType>;
    auto GTextractFilter = ExtractFilterType::New();
    GTextractFilter->SetDirectionCollapseToSubmatrix();
    auto PdextractFilter = ExtractFilterType::New();
    PdextractFilter->SetDirectionCollapseToSubmatrix();

    //get regions to compare
    GTImageType::RegionType GTregion = GTinput->GetLargestPossibleRegion();
    GTImageType::SizeType   GTsize = GTregion.GetSize();
    PdImageType::RegionType Pdregion = Pdinput->GetLargestPossibleRegion();
    PdImageType::SizeType   Pdsize = Pdregion.GetSize();

    // set up the extraction region [one slice]
    //GTsize[2] = 1; // extract along z direction
    //Pdsize[2] = 1; // extract along z direction
    GTImageType::IndexType GTstart = GTregion.GetIndex();
    PdImageType::IndexType Pdstart = Pdregion.GetIndex();

    GTImageType::RegionType GTdesiredRegion;
    GTdesiredRegion.SetSize(GTsize);

    GTImageType::RegionType PddesiredRegion;
    PddesiredRegion.SetSize(Pdsize);
    int size = GTsize[2];


//error here.. not sure why
    // begin at the top of the GT
    for (int i = 0; i < size; i++) {
        std::cout << i << "," << GTsize[2] << std::endl;
        unsigned int sliceNumber = i;

        GTstart[2] = sliceNumber;
        //GTImageType::RegionType GTdesiredRegion;
        //GTdesiredRegion.SetSize(GTsize);
        GTdesiredRegion.SetIndex(GTstart);

        // begin at the top of the predicted
        Pdstart[2] = sliceNumber;
        //GTImageType::RegionType PddesiredRegion;
        //PddesiredRegion.SetSize(Pdsize);
        PddesiredRegion.SetIndex(Pdstart);

        using SliceType = itk::Image<unsigned char, 2>;
        //start
        GTextractFilter->SetExtractionRegion(GTdesiredRegion);
        GTextractFilter->SetInput(GTinput);
        //Pdslice = SetInput(PdextractFilter->GetOutput())

        PdextractFilter->SetExtractionRegion(PddesiredRegion);
        PdextractFilter->SetInput(Pdinput);
        //Pdslice = SetInput(PdextractFilter->GetOutput())

        //normalize the slice
        using NormalizeFilterType = itk::NormalizeImageFilter<GTImageType, GTRescaleImageType>;
        auto GTNormalizer = NormalizeFilterType::New();
        auto PdNormalizer = NormalizeFilterType::New();

        GTNormalizer->SetInput(GTextractFilter->GetOutput());
        PdNormalizer->SetInput(PdextractFilter->GetOutput());
        GTNormalizer->Update();
        PdNormalizer->Update();


        //Smooth the slice
        using GaussianFilterType = itk::DiscreteGaussianImageFilter<GTImageType, GTRescaleImageType>;

        auto GTSmoother = GaussianFilterType::New();
        auto PdSmoother = GaussianFilterType::New();

        GTSmoother->SetVariance(2.0);
        PdSmoother->SetVariance(2.0);

        GTSmoother->SetInput(GTNormalizer->GetOutput());
        PdSmoother->SetInput(PdNormalizer->GetOutput());
        GTSmoother->Update();
        PdSmoother->Update();

        //calculate the histograms
        //using SubtractImageFilterType = itk::SubtractImageFilter<GTImageType, GTRescaleImageType>;

        //auto subtractFilter = SubtractImageFilterType::New();
        //subtractFilter->SetInput1(PdSmoother->GetOutput());
        //subtractFilter->SetInput2(GTSmoother->GetOutput());
        //subtractFilter->Update();
        //std::cout << "subtraction value is: " << subtractFilter->GetOutput() << std::endl;

        constexpr unsigned int MeasurementVectorSize = 1; // Grayscale
        const auto             binsPerDimension = static_cast<unsigned int>(std::stoi(argv[3]));

        using ImageToHistogramFilterType = itk::Statistics::ImageToHistogramFilter<GTImageType>;

        ImageToHistogramFilterType::HistogramType::MeasurementVectorType lowerBound(binsPerDimension);
        lowerBound.Fill(0);

        ImageToHistogramFilterType::HistogramType::MeasurementVectorType upperBound(binsPerDimension);
        upperBound.Fill(255);

        ImageToHistogramFilterType::HistogramType::SizeType size(MeasurementVectorSize);
        size.Fill(binsPerDimension);

        auto imageToHistogramFilter = ImageToHistogramFilterType::New();
        imageToHistogramFilter->SetInput(GTSmoother->GetOutput());
        imageToHistogramFilter->SetHistogramBinMinimum(lowerBound);
        imageToHistogramFilter->SetHistogramBinMaximum(upperBound);
        imageToHistogramFilter->SetHistogramSize(size);

        try
        {
            imageToHistogramFilter->Update();
        }
        catch (const itk::ExceptionObject& error)
        {
            std::cerr << "Error: " << error << std::endl;
            return EXIT_FAILURE;
        }

        ImageToHistogramFilterType::HistogramType* histogram = imageToHistogramFilter->GetOutput();

        std::cout << "The histogram for the ground truth is: " << std::endl;
        std::cout << "Frequency = [ ";
        for (unsigned int i = 0; i < histogram->GetSize()[0]; ++i)
        {
            std::cout << histogram->GetFrequency(i);

            if (i != histogram->GetSize()[0] - 1)
            {
                std::cout << ",";
            }
        }

        std::cout << " ]" << std::endl;

        imageToHistogramFilter->SetInput(PdSmoother->GetOutput());
        imageToHistogramFilter->SetHistogramBinMinimum(lowerBound);
        imageToHistogramFilter->SetHistogramBinMaximum(upperBound);
        imageToHistogramFilter->SetHistogramSize(size);

        try
        {
            imageToHistogramFilter->Update();
        }
        catch (const itk::ExceptionObject& error)
        {
            std::cerr << "Error: " << error << std::endl;
            return EXIT_FAILURE;
        }

        std::cout << "The histogram for the predicted is: " << std::endl;
        std::cout << "Frequency = [ ";
        for (unsigned int i = 0; i < histogram->GetSize()[0]; ++i)
        {
            std::cout << histogram->GetFrequency(i);

            if (i != histogram->GetSize()[0] - 1)
            {
                std::cout << ",";
            }
        }

        std::cout << " ]" << std::endl;
    }
    return EXIT_SUCCESS;
}
