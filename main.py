

import cv2
import os
import logging
import time
import numpy as np
from face_detection import Model_Face_detection
from facial_landmarks_detection import Model_Facial_landmark
from gaze_estimation import Model_Gaze_estimation
from head_pose_estimation import Model_Head_pose
from mouse_controller import MouseController
import argparse
from input_feeder import InputFeeder

def build_argparser():
    #Parse command line arguments.

    #return arguments
    parser=argparse.ArgumentParser()
    parser.add_argument("-f", "--facedetectionmodel", required=True, type=str,
                        help="Specify Path to .xml file of Face Detection model.")
    parser.add_argument("-fl", "--faciallandmarkmodel", required=True, type=str,
                        help="Specify Path to .xml file of Facial Landmark Detection model.")
    parser.add_argument("-hp", "--headposemodel", required=True, type=str,
                        help="Specify Path to .xml file of Head Pose Estimation model.")
    parser.add_argument("-g", "--gazeestimationmodel", required=True, type=str,
                        help="Specify Path to .xml file of Gaze Estimation model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Specify Path to video file or enter cam for webcam")
    parser.add_argument("-flags", "--previewFlags", required=False, nargs='+',
                        default=[],
                        help="Specify the flags from fd, fld, hp, ge like --flags fd hp fld (Seperate each flag by space)"
                             "for see the visualization of different model outputs of each frame," 
                             "fd for Face Detection, fld for Facial Landmark Detection"
                             "hp for Head Pose Estimation, ge for Gaze Estimation." )
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-prob", "--prob_threshold", required=False, type=float,
                        default=0.6,
                        help="Probability threshold for model to detect the face accurately from the video frame.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-o", '--output_path', default='C:/Users/jiaquan1/Desktop/Intel_Edge_AI_IOT_NanoProgram/Models/results/', type=str)
    
    args=parser.parse_args()
    return args

def draw_preview(
        frame, preview_flags, cropped_image, left_eye_image, right_eye_image,
        face_cords, eye_cords, pose_output, gaze_vector):
    preview_frame = frame.copy()

    if 'ff' in preview_flags:
        if len(preview_flags) != 1:
            preview_frame = cropped_image
        cv2.rectangle(frame, (face_cords[0], face_cords[1]), (face_cords[2], face_cords[3]),
                      (0, 0, 0), 3)

    if 'fl' in preview_flags:
        cv2.rectangle(cropped_image, (eye_cords[0][0]-10, eye_cords[0][1]-10), (eye_cords[0][2]+10, eye_cords[0][3]+10),
                      (255, 0, 0), 2)
        cv2.rectangle(cropped_image, (eye_cords[1][0]-10, eye_cords[1][1]-10), (eye_cords[1][2]+10, eye_cords[1][3]+10),
                      (255, 0, 0), 2)

    if 'fh' in preview_flags:
        cv2.putText(
            frame,
            "Pose Angles: yaw= {:.2f} , pitch= {:.2f} , roll= {:.2f}".format(
                pose_output[0], pose_output[1], pose_output[2]),
            (20, 40),
            cv2.FONT_HERSHEY_COMPLEX,
            1, (0, 0, 0), 2)

    if 'fg' in preview_flags:

        cv2.putText(
            frame,
            "Gaze Cords: x= {:.2f} , y= {:.2f} , z= {:.2f}".format(
                gaze_vector[0], gaze_vector[1], gaze_vector[2]),
            (20, 80),
            cv2.FONT_HERSHEY_COMPLEX,
            1, (0, 0, 0), 2)

        x, y, w = int(gaze_vector[0] * 12), int(gaze_vector[1] * 12), 160
        le = cv2.line(left_eye_image.copy(), (x - w, y - w), (x + w, y + w), (255, 0, 255), 2)
        cv2.line(le, (x - w, y + w), (x + w, y - w), (255, 0, 255), 2)
        re = cv2.line(right_eye_image.copy(), (x - w, y - w), (x + w, y + w), (255, 0, 255), 2)
        cv2.line(re, (x - w, y + w), (x + w, y - w), (255, 0, 255), 2)
        preview_frame[eye_cords[0][1]:eye_cords[0][3], eye_cords[0][0]:eye_cords[0][2]] = le
        preview_frame[eye_cords[1][1]:eye_cords[1][3], eye_cords[1][0]:eye_cords[1][2]] = re

    return preview_frame


def infer_on_stream(args):

    # Grab args from command line
    preview_flags = args.previewFlags
    
    logger = logging.getLogger()
    inputFilePath = args.input
    output_path = args.output_path
    inputFeeder = None
    if inputFilePath.lower()=="cam":
            inputFeeder = InputFeeder("cam")
    else:
        if not os.path.isfile(inputFilePath):
            print(inputFilePath)
            logger.error("Unable to find specified video file")
            exit(1)
        inputFeeder = InputFeeder("video",inputFilePath)
    
    modelPathDict = {'FaceDetectionModel':args.facedetectionmodel, 'FacialLandmarksDetectionModel':args.faciallandmarkmodel, 
    'GazeEstimationModel':args.gazeestimationmodel, 'HeadPoseEstimationModel':args.headposemodel}
    
    for fileNameKey in modelPathDict.keys():
        if not os.path.isfile(modelPathDict[fileNameKey]):
            logger.error("Unable to find specified "+fileNameKey+" xml file")
            exit(1)
            
    fdm = Model_Face_detection(modelPathDict['FaceDetectionModel'], args.device, args.cpu_extension)
    fldm = Model_Facial_landmark(modelPathDict['FacialLandmarksDetectionModel'], args.device, args.cpu_extension)
    gem = Model_Gaze_estimation(modelPathDict['GazeEstimationModel'], args.device, args.cpu_extension)
    hpem = Model_Head_pose(modelPathDict['HeadPoseEstimationModel'], args.device, args.cpu_extension)
    
    mc = MouseController('medium','fast')
    
    inputFeeder.load_data()
    start_model_load_time = time.time()
    fdm.load_model()
    fldm.load_model()
    hpem.load_model()
    gem.load_model()
    total_model_load_time = time.time() - start_model_load_time

    out_video = cv2.VideoWriter(os.path.join('output_video.mp4'),cv2.VideoWriter_fourcc(*"avc1"), int(inputFeeder.get_fps()/10),(1920, 1080), True)
    
    frame_count = 0
    start_inference_time = time.time()
    for ret, frame in inputFeeder.next_batch():
        if not ret:
            break
        frame_count+=1
        # if frame_count%5==0:
        #     cv2.imshow('video',cv2.resize(frame,(500,500)))
    
        key = cv2.waitKey(60)
        croppedFace, face_coords = fdm.predict(frame.copy(),args.prob_threshold)
        if type(croppedFace)==int:
            logger.error("Unable to detect the face.")
            if key==27:
                break
            continue
        
        hp_out = hpem.predict(croppedFace.copy())
        
        left_eye, right_eye, eye_coords = fldm.predict(croppedFace.copy())
        
        new_mouse_coord, gaze_vector = gem.predict(left_eye, right_eye, hp_out)
        
       

        if not len(preview_flags) == 0:
            preview_frame = draw_preview(
                frame, preview_flags, croppedFace, left_eye, right_eye,
                face_coords, eye_coords, hp_out, gaze_vector)

        
        image = np.hstack((cv2.resize(frame, (500, 500)), cv2.resize(preview_frame, (500, 500))))
        cv2.imshow('preview', image)
        out_video.write(frame)
        
            
        if frame_count%5==0:
            mc.move(new_mouse_coord[0],new_mouse_coord[1])    
        if key==27:
                break
    total_time = time.time() - start_inference_time
    total_inference_time = round(total_time, 1)
    fps = frame_count / total_inference_time

    try:
        os.mkdir(output_path)
    except OSError as error:
         logger.info("Result file created")

    with open(output_path+'stats.txt', 'w') as f:
        f.write(str(total_inference_time) + '\n')
        f.write(str(fps) + '\n')
        f.write(str(total_model_load_time) + '\n')

    logger.error('Model load time: ' + str(total_model_load_time))
    logger.error('Inference time: ' + str(total_inference_time))
    logger.error('FPS: ' + str(fps))
    logger.error("VideoStream ended...")
    cv2.destroyAllWindows()
    inputFeeder.close()
    
def main():
    args = build_argparser()
    infer_on_stream(args)    

if __name__ == '__main__':
    main() 