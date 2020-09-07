# Computer Pointer Controller

Introduction
In this project, I used face detection model, head pose estimation model, face landmark detection model and gaze detection model to control the mouse pointer on my computer.  I used the Gaze Estimation model to estimate the gaze of the user's eyes and change the mouse pointer position accordingly. This project demonstrated my ability to run multiple models in the same machine and coordinate the flow of data between those models.

I used the InferenceEngine API from Intel's OpenVino ToolKit to build the project. This project is for the Intel@ Edge AI FOR IoT Developers Nanodegree Program.
Four models used in this project:
1. face detection model (https://docs.openvinotoolkit.org/latest/omz_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)
2. head pose estimation model (https://docs.openvinotoolkit.org/latest/omz_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
3. face landmark detection model (https://docs.openvinotoolkit.org/latest/omz_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)
4. gaze detection model (https://docs.openvinotoolkit.org/latest/omz_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)

The Pipeline:
In this project I coordinated the flow of data from the input, and then amongst the different models and finally to the mouse controller. The flow of data look like this:

![image](https://github.com/jiaquan1/computer_pointer_controller/blob/master/bin/pipeline.png)

Project Set Up and Installation

Step1: Download below three softwares:

Microsoft Visual Studio* with C++ 2019, 2017, or 2015 with MSBuild
CMake 3.4 or higher 64-bit NOTE: If you want to use Microsoft Visual Studio 2019, you are required to install CMake 3.14.
Python 3.6.5 64-bit
Step2. Download OpenVino Toolkit 2020.1 with all the prerequisites by following this installation guide

Step3: Setup OpenVino Toolkit using below command in command prompt

cd C:\Program Files (x86)\IntelSWTools\openvino\bin\
setupvars.bat
Step4: Configure Model Optimizer using below commnads in command prompt

cd C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\model_optimizer\install_prerequisites
install_prerequisites.bat
Step5: Varify installation

cd C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\demo\
demo_squeezenet_download_convert_run.bat
Above command should give output like this image optimizer_output:

Demo
Step1. Clone the Repository using git clone https://github.com/jiaquan1/computer_pointer_controller.git
Step2. Instantiate OpenVino Environment. For windows use below command

cd C:\Program Files (x86)\IntelSWTools\openvino\bin\
setupvars.bat

Step3. Go back to the project directory folder
cd path_of_project_directory

Step4. Run below commands to execute the project

python main.py -fd /face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml \ 
-lr /landmarks-regression-retail-0009/FP32-INT8/landmarks-regression-retail-0009.xml \ 
-hp /head-pose-estimation-adas-0001/FP32-INT8/head-pose-estimation-adas-0001.xml \ 
-ge /gaze-estimation-adas-0002/FP32-INT8/gaze-estimation-adas-0002.xml \ 
-i /bin/demo.mp4 -flags ff fl fh fg

Step5. To run it on GPU, run below commands to execute:
python main.py -fd /face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml \ 
-lr /landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml \ 
-hp /head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml \ 
-ge /gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.xml \ 
-i /bin/demo.mp4 -flags ff fl fh fg -d GPU

Documentation
Command Line Argument Information:

fd : Specify path of xml file of face detection model
lr : Specify path of xml file of landmark regression model
hp : Specify path of xml file of Head Pose Estimation model
ge : Specify path of xml file of Gaze Estimation model
i : Specify path of input Video file or cam for Webcam
flags (Optional): if you want to see preview video in separate window you need to Specify flag from ff, fl, fh, fg like -flags ff fl...(Space seperated if multiple values) ff for faceDetectionModel, fl for landmarkRegressionModel, fh for headPoseEstimationModel, fg for gazeEstimationModel
probs (Optional): if you want to specify confidence threshold for face detection, you can specify the value here in range(0, 1), default=0.6
d (Optional): Specify Device for inference, the device can be CPU, GPU, FPGU, MYRID
o : Specify path of output folder where we will store results
## Benchmarks
I tested on my local computer on CPU (Intel(R) Core(TM) i7-8665U CPU @1.90GHz 2.11GHz) and GPU (Intel® UHD Graphics 620). I have checked Inference Time, Model Loading Time, and Frames Per Second model for FP16, FP32, and FP16-INT8 of all the models except Face Detection Model. Face Detection Model was only available on FP32-INT1 precision. You can use below commands to get results for respective precisions:
FP16:
python main.py -fd /face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml \ 
-lr /landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml \ 
-hp /head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml \ 
-ge /gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml \ 
-i /bin/demo.mp4 -flags ff fl fh fg
INT8:
python main.py -fd /face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml \ 
-lr /landmarks-regression-retail-0009/FP16-INT8/landmarks-regression-retail-0009.xml \ 
-hp /head-pose-estimation-adas-0001/FP16-INT8/head-pose-estimation-adas-0001.xml \ 
-ge /gaze-estimation-adas-0002/FP16-INT8/gaze-estimation-adas-0002.xml \ 
-i /bin/demo.mp4 -flags ff fl fh fg

Project folder contains all the executable files:

face_detection.py

Contains preprocession of video frame, perform infernce on it and detect the face, postprocess the outputs.
facial_landmarks_detection.py

Take the deteted face as input, preprocessed it, perform inference on it and detect the eye landmarks, postprocess the outputs.
head_pose_estimation.py

Take the detected face as input, preprocessed it, perform inference on it and detect the head postion by predicting yaw - roll - pitch angles, postprocess the outputs.
gaze_estimation.py

Take the left eye, rigt eye, head pose angles as inputs, preprocessed it, perform inference and predict the gaze vector, postprocess the outputs.
input_feeder.py

Contains InputFeeder class which initialize VideoCapture as per the user argument and return the frames one by one.
mouse_controller.py

Contains MouseController class which take x, y coordinates value, speed, precisions and according these values it moves the mouse pointer by using pyautogui library.
main.py

Users need to run main.py file for running the app.
bin folder contains the demo.mp4 files
results folder is to save the statistical result from the running.
Other 4 folders are the model files downed from open zoo. 


Benchmark results of running my model on multiple hardwares and multiple model precisions. My benchmarks can include: model loading time, input/output processing time, model inference time.

Device	Precision model_time(s)	Inference_Time(s) FPS
CPU	FP32	1.013162851	24.6	2.398373984
CPU	FP16	1.034621477	24.4	2.418032787
CPU	INT8	1.17280674	25.7	2.295719844
GPU	FP32	57.16926455	26.6	2.218045113
GPU	FP16	58.18735433	26.4	2.234848485
GPU	INT8	67.42535996	31.2	1.891025641

Results
For both CPU and GPU, FP32 has smallest model time.
For both CPU and GPU, FP16 has smallest inference time, this can be said as combination of precisions lead to higher weight of the model.
For both CPU and GPU, FP16 has slightly better performance regarding FPS, this is coincidence with the inference time result. 
CPU has much smaller model time, and less interence time than GPU at all precision level, this might because the GPU on this device is not optimized to to run these models. Many layers are not supported on GPU, there is much data traffic between GPU and CPU when running on GPU, which slowed down the process. 

Stand Out Suggestions
Edge Cases
Multiple People Scenario: If we encounter multiple people in the video frame, it will always use and give results one face even though multiple people detected,
No Head Detection: it will skip the frame and inform the user
