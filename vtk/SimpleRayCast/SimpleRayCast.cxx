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

#include <iostream>
#include <fstream>
#include <string>
#include <cstring>

#include "utility.hpp"
#include "cmap.hpp"

#define CTF_TEST()\
	colorTransferFunction->AddRGBPoint(0.0, 1.0, 1.0, 1.0);\
	colorTransferFunction->AddRGBPoint(1.0, 0.0, 0.0, 0.0);


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
int main(int argc, char* argv[])
{
	vtkMultiThreader::SetGlobalMaximumNumberOfThreads(1);
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

 	vtkNew<vtkRenderWindowInteractor> iren;
  	iren->SetRenderWindow(renWin);

  	// Create the reader for the data
  	vtkNew<vtkStructuredPointsReader> reader;
  	reader->SetFileName(argv[1]);
  	reader->Update();

  	double r[2];
  	reader->GetOutput()->GetScalarRange(r);
  	std::cout << "min: " << r[0] << std::endl;
  	std::cout << "max: " << r[1] << std::endl;

	float opacity_start_value = std::stof(argv[11]);	
  	// Create transfer mapping scalar value to opacity
  	// vtkNew<vtkPiecewiseFunction> opacityTransferFunction;
  	vtkSmartPointer<vtkPiecewiseFunction> opacityTransferFunction = vtkSmartPointer<vtkPiecewiseFunction>::New();
	opacityTransferFunction->AddPoint(0.0, 0.0);
  	opacityTransferFunction->AddPoint(opacity_start_value, 0.0);
  	opacityTransferFunction->AddPoint(1.0, 1.0);

  	// Create transfer mapping scalar value to color
  	vtkNew<vtkColorTransferFunction> colorTransferFunction;
	CTF_TEST();

  	// The property describes how the data will look
  	vtkNew<vtkVolumeProperty> volumeProperty;
  	volumeProperty->SetColor(colorTransferFunction);
  	volumeProperty->SetScalarOpacity(opacityTransferFunction);

	// Set lighting
#if 1
  	// volumeProperty->ShadeOn();
	// volumeProperty->SetAmbient(0.0);  // Ambient coefficient (default: 0.0)
	volumeProperty->SetDiffuse(1.0);  // Diffuse coefficient (default: 0.0)
	volumeProperty->SetSpecular(1.0); // Specular coefficient (default: 0.0)
	// volumeProperty->SetSpecularPower(10.0); // Controls the sharpness of specular highlights
	// vtkSmartPointer<vtkLight> light = vtkSmartPointer<vtkLight>::New();
	// light->SetPosition(0.0, 1.0, 0.0); // Set light position
	// light->SetIntensity(1);          // Set light intensity
	// ren1->AddLight(light);  	
#endif

	volumeProperty->SetInterpolationTypeToLinear();

  	// The mapper / ray cast function know how to render the data
  	vtkNew<vtkFixedPointVolumeRayCastMapper> volumeMapper;
  	// vtkNew<vtkGPUVolumeRayCastMapper> volumeMapper; // irrelevant to sample distance
  	// vtkNew<vtkOpenGLGPUVolumeRayCastMapper> volumeMapper; // irrelevant to sample distance
  	// vtkNew<vtkSmartVolumeMapper> volumeMapper; // irrelevant to sample distance
  	volumeMapper->SetInputConnection(reader->GetOutputPort());
  	std::cout << "sample distance: " << volumeMapper->GetSampleDistance() << std::endl;
  	volumeMapper->SetSampleDistance(0.1); // 0.1 works best for 256 cube data with 1 spacing
  	// volumeMapper->SetSampleDistance(0.01); // 0.01 only used for timing test
  	std::cout << "sample distance: " << volumeMapper->GetSampleDistance() << std::endl;
  	volumeMapper->SetBlendModeToComposite();

  	// The volume holds the mapper and the property and
  	// can be used to position/orient the volume
  	vtkNew<vtkVolume> volume;
  	volume->SetMapper(volumeMapper);
  	volume->SetProperty(volumeProperty);

	// turnOnAxis(volume, ren1);
	// turnOnBoundingBox(reader, ren1);

  	ren1->AddVolume(volume);
  	// ren1->SetBackground(colors->GetColor3d("Wheat").GetData());
  	// ren1->SetBackground(0, 0, 0);
  	ren1->SetBackground(1, 1, 1);
  	// ren1->GetActiveCamera()->Azimuth(45);
  	// ren1->GetActiveCamera()->Elevation(30);

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

  	// iren->Start();

  	return EXIT_SUCCESS;
}
