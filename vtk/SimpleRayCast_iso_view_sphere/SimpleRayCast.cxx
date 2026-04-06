#include <vtkCamera.h>
#include <vtkColorTransferFunction.h>
#include <vtkFixedPointVolumeRayCastMapper.h>
#include <vtkNamedColors.h>
#include <vtkNew.h>
#include <vtkPiecewiseFunction.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkStructuredPointsReader.h>
#include <vtkVolume.h>
#include <vtkVolumeProperty.h>
#include <vtkPolyDataMapper.h>
#include <vtkPNGWriter.h>
#include <vtkWindowToImageFilter.h>
#include <vtkMultiThreader.h>
#include <vtkContourFilter.h>
#include <vtkProperty.h>

vtkSmartPointer<vtkActor>
CreateIsosurfaceActor(vtkAlgorithmOutput* input,
                      double isoValue,
                      double opacity,
                      double color[3])
{
    auto contour = vtkSmartPointer<vtkContourFilter>::New();
    contour->SetInputConnection(input);
    contour->SetValue(0, isoValue);
    contour->Update();

    auto mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(contour->GetOutputPort());
    mapper->ScalarVisibilityOff();

    auto actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    actor->GetProperty()->SetColor(color);
    actor->GetProperty()->SetOpacity(opacity);

    actor->GetProperty()->SetAmbient(0.1);
    actor->GetProperty()->SetDiffuse(0.7);
    actor->GetProperty()->SetSpecular(0.6);
    actor->GetProperty()->SetSpecularPower(80);
    actor->GetProperty()->BackfaceCullingOff();

    return actor;
}

int main(int argc, char* argv[])
{
	vtkMultiThreader::SetGlobalMaximumNumberOfThreads(0);
	if (argc < 2)
	{
    	std::cout << "Usage: " << argv[0] << " ironProt.vtk" << std::endl;
    	return EXIT_FAILURE;
	}

	// Create the standard renderer, render window and interactor
  	vtkNew<vtkNamedColors> colors;
  	vtkNew<vtkRenderer> ren1;
  	vtkNew<vtkRenderWindow> renWin;
 	renWin->AddRenderer(ren1);

	// Recommended for transparency correctness
	renWin->SetAlphaBitPlanes(1);
	renWin->SetMultiSamples(0);
	ren1->SetUseDepthPeeling(1);
	ren1->SetMaximumNumberOfPeels(100);
	ren1->SetOcclusionRatio(0.1);

 	vtkNew<vtkRenderWindowInteractor> iren;
  	iren->SetRenderWindow(renWin);

  	// Create the reader for the data
  	vtkNew<vtkStructuredPointsReader> reader;
  	reader->SetFileName(argv[1]);
  	reader->Update();

	// double color1[3] = {0.0, 1.0, 0.0}; // green
	double color1[3] = {0.65, 0.85, 0.70}; // green
	double color2[3] = {1.0, 1.0, 1.0}; // white

	// auto iso1 = CreateIsosurfaceActor(reader->GetOutputPort(), 0.12, 0.1, color1);
	auto iso1 = CreateIsosurfaceActor(reader->GetOutputPort(), 0.12, 0.45, color1);
	auto iso2 = CreateIsosurfaceActor(reader->GetOutputPort(), 0.45, 0.99, color2);

	ren1->AddActor(iso1);
	ren1->AddActor(iso2);
  
  	ren1->SetBackground(1, 1, 1);

  	ren1->ResetCameraClippingRange();
  	ren1->ResetCamera();
	
	// set camera parameters from arguments
	double position_x = std::stod(argv[2]);
	double position_y = std::stod(argv[3]);
	double position_z = std::stod(argv[4]);
	double focal_point_x = std::stod(argv[5]);
	double focal_point_y = std::stod(argv[6]);
	double focal_point_z = std::stod(argv[7]);
	double view_up_x = std::stod(argv[8]);
	double view_up_y = std::stod(argv[9]);
	double view_up_z = std::stod(argv[10]);

  	ren1->GetActiveCamera()->SetPosition(position_x,
										 position_y,
										 position_z);
  	ren1->GetActiveCamera()->SetFocalPoint(focal_point_x,
										   focal_point_y,
										   focal_point_z);
  	ren1->GetActiveCamera()->SetViewUp(view_up_x,
									   view_up_y,
									   view_up_z);
  	ren1->GetActiveCamera()->SetViewAngle(30);
  	ren1->GetActiveCamera()->SetClippingRange(0.1, 1000);
  
  	// renWin->SetSize(1024, 1024);
  	renWin->SetSize(256, 256);
  	// renWin->SetSize(512, 512);
  	renWin->Render();

    // Screenshot
    vtkNew<vtkWindowToImageFilter> windowToImageFilter;
    windowToImageFilter->SetInput(iren->GetRenderWindow());
    windowToImageFilter->SetScale(1); // image quality
    windowToImageFilter->SetInputBufferTypeToRGB(); // also record the alpha
    windowToImageFilter->ReadFrontBufferOff();       // read from the back buffer
    vtkNew<vtkPNGWriter> writer;
    std::string fileName = "rendering.png";
    writer->SetFileName(fileName.c_str());
    writer->SetInputConnection(windowToImageFilter->GetOutputPort());
    writer->Write();

  	// iren->Start(); // comment out for rendering one frame and self closing this rendering app

  	return EXIT_SUCCESS;
}
