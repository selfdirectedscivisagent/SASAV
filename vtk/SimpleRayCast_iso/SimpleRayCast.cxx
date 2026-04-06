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

#include <vtkStructuredPoints.h>
#include <vtkGPUVolumeRayCastMapper.h>
#include <vtkSmartVolumeMapper.h>
#include <vtkOpenGLGPUVolumeRayCastMapper.h>

#include <vtkTransform.h>
#include <vtkAxesActor.h>
#include <vtkTextActor.h>
#include <vtkTextProperty.h>
#include <vtkCaptionActor2D.h>
#include <vtkOutlineFilter.h>
#include <vtkPolyDataMapper.h>
#include <vtkPNGWriter.h>
#include <vtkWindowToImageFilter.h>
#include <vtkMultiThreader.h>
#include <vtkColorSeries.h>
#include <vtkLight.h>
#include <vtkContourFilter.h>
#include <vtkProperty.h>

#include <vtkFlyingEdges3D.h>
#include <vtkWindowedSincPolyDataFilter.h>
#include <vtkPolyDataNormals.h>
#include <vtkImageResample.h>

#include <iostream>
#include <fstream>
#include <string>
#include <cstring>

#include "utility.hpp"
#include "cmap.hpp"

int view_size = 0;
std::vector<std::vector<double>> view_list;

class vtkTimerCallback2 : public vtkCommand
{
public:
  vtkTimerCallback2() = default;
  ~vtkTimerCallback2() = default;

  int timerId = 0;
  static vtkTimerCallback2* New()
  {
    vtkTimerCallback2* cb = new vtkTimerCallback2;
    cb->TimerCount = 0;
    return cb;
  }
  virtual void Execute(vtkObject* caller, unsigned long eventId,
                       void* vtkNotUsed(callData))
  {
    vtkRenderWindowInteractor* iren =
        dynamic_cast<vtkRenderWindowInteractor*>(caller);
    if (vtkCommand::TimerEvent == eventId)
    {
      ++this->TimerCount;
    }
    std::cout << "TimerCount: " << TimerCount << std::endl; //TimerCount starts from 1
	std::cout << "view_size: "  << view_size << std::endl;
	std::cout << "view_list size: " << view_list.size() << std::endl;
#if 1
    if (TimerCount < view_size + 1) // image index from 1 to view_size
    // if (TimerCount < 11) // image index from 1 to view_size
    {
		camera->SetPosition(view_list.at(TimerCount - 1).at(0),
						    view_list.at(TimerCount - 1).at(1),
							view_list.at(TimerCount - 1).at(2));
		camera->SetFocalPoint(view_list.at(TimerCount - 1).at(3),
						      view_list.at(TimerCount - 1).at(4),
							  view_list.at(TimerCount - 1).at(5));
		camera->SetViewUp(view_list.at(TimerCount - 1).at(6),
						  view_list.at(TimerCount - 1).at(7),
					      view_list.at(TimerCount - 1).at(8));
		printCamera(camera);
#if 1 
      // Screenshot
      vtkNew<vtkWindowToImageFilter> windowToImageFilter;
      windowToImageFilter->SetInput(iren->GetRenderWindow());
      windowToImageFilter->SetScale(1); // image quality
      // windowToImageFilter->SetInputBufferTypeToRGBA(); // also record the alpha
      windowToImageFilter->SetInputBufferTypeToRGB(); // also record the alpha
      windowToImageFilter->ReadFrontBufferOff();       // read from the back buffer
      // windowToImageFilter->Update();

      vtkNew<vtkPNGWriter> writer;
      std::string fileName = "screenShot/" + std::to_string(TimerCount) + ".png";
      writer->SetFileName(fileName.c_str());
      writer->SetInputConnection(windowToImageFilter->GetOutputPort());
      writer->Write();
      std::cout << "=======Saving Done!" << endl;
#endif
      // iren->GetRenderWindow()->Render();

      // iren->GetRenderWindow()->Render();
    }
    else
    {
      std::cout << "Timer distroied" << std::endl;
      iren->DestroyTimer();
    }
#endif
  }

private:
  int TimerCount = 0;

public:
  // vtkActor* actor;
  vtkVolume* volume;
  vtkCamera* camera;
};

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

    // Lighting (important for transparent surfaces)
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

	// --- Recommended for transparency correctness ---
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

#if 0
	// Create the isosurface
    vtkSmartPointer<vtkContourFilter> contourFilter = vtkSmartPointer<vtkContourFilter>::New();
    contourFilter->SetInputConnection(reader->GetOutputPort());
    contourFilter->SetValue(0, 0.12); // Set the isosurface value to 0.12
    contourFilter->Update();
	// Create a mapper and actor for the isosurface
    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(contourFilter->GetOutputPort());
    mapper->ScalarVisibilityOff();
	vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);
	// Enable shading and set specular properties
    actor->GetProperty()->SetAmbient(0.1);  // Ambient lighting contribution
    actor->GetProperty()->SetDiffuse(0.7);  // Diffuse lighting contribution
    actor->GetProperty()->SetSpecular(1.0); // Specular lighting contribution (set to 1)
    actor->GetProperty()->SetSpecularPower(100); // Controls the sharpness of specular highlights
    actor->GetProperty()->SetSpecularColor(1.0, 1.0, 1.0); // Set specular color to white
	actor->GetProperty()->SetOpacity(0.3);
	ren1->AddActor(actor);
#endif

	double color1[3] = {1.0, 0.0, 0.0}; // red
	double color2[3] = {0.0, 1.0, 0.0}; // green
	double color3[3] = {0.0, 0.0, 1.0}; // blue
	double color4[3] = {1.0, 1.0, 1.0}; // white

	float opacity_start_value = std::stof(argv[11]);	
	//auto iso1 = CreateIsosurfaceActor(
    // 	reader->GetOutputPort(), 0.12, 0.1, color2);

	auto iso2 = CreateIsosurfaceActor(reader->GetOutputPort(), opacity_start_value, 1.0, color4);

	// auto iso3 = CreateIsosurfaceActor(
    // 	reader->GetOutputPort(), 0.20, 0.6, color3);

	// ren1->AddActor(iso1);

	ren1->AddActor(iso2);
	// ren1->AddActor(iso3);
  	
  
  	// ren1->SetBackground(0, 0, 0);
  	ren1->SetBackground(1, 1, 1);

  	ren1->ResetCameraClippingRange();
  	ren1->ResetCamera();
  	printCamera(ren1->GetActiveCamera());
	// /*
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
	// */
  	ren1->GetActiveCamera()->SetViewAngle(30);
  	ren1->GetActiveCamera()->SetClippingRange(0.1, 1000);
  	printCamera(ren1->GetActiveCamera());
  
  	// renWin->SetSize(1024, 1024);
  	// renWin->SetSize(512, 512);
  	renWin->SetSize(256, 256);
  	// renWin->SetSize(64, 64);
  	renWin->SetWindowName("Rendering Result");
  	renWin->Render();

#if 1 
      // Screenshot
      vtkNew<vtkWindowToImageFilter> windowToImageFilter;
      windowToImageFilter->SetInput(iren->GetRenderWindow());
      windowToImageFilter->SetScale(1); // image quality
      // windowToImageFilter->SetInputBufferTypeToRGBA(); // also record the alpha
      windowToImageFilter->SetInputBufferTypeToRGB(); // also record the alpha
      windowToImageFilter->ReadFrontBufferOff();       // read from the back buffer
      // windowToImageFilter->Update();

      vtkNew<vtkPNGWriter> writer;
      std::string fileName = "rendering.png";
      writer->SetFileName(fileName.c_str());
      writer->SetInputConnection(windowToImageFilter->GetOutputPort());
      writer->Write();
      std::cout << "=======Saving rendering Done!" << endl;
#endif


#if 0 // starts animation
    // Initialize must be called prior to creating timer events.
    iren->Initialize();
    // Sign up to receive TimerEvent
    vtkNew<vtkTimerCallback2> cb;
    cb->volume = volume;
	cb->camera = ren1->GetActiveCamera();
    iren->AddObserver(vtkCommand::TimerEvent, cb);
    int timerId = iren->CreateRepeatingTimer(1000); // ms
    cb->timerId = timerId;
#endif

  	// iren->Start(); // comment out for rendering one frame and self closing this rendering app

  	return EXIT_SUCCESS;
}
