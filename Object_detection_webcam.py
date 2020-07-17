######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/20/18
# Description: 
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier uses it to perform object detection on a webcam feed.
# It draws boxes and scores around the objects of interest in each frame from
# the webcam.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.


# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import time
import logging
from networktables import NetworkTables


# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'



# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 1

## Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')
setHeight = 480
setWidth = 640
# Initialize webcam feed
video = cv2.VideoCapture(0)
ret = video.set(3, setWidth) #width
ret = video.set(4, setHeight) #height


#video.set() commands
# 0. CV_CAP_PROP_POS_MSEC Current position of the video file in milliseconds.
# 1. CV_CAP_PROP_POS_FRAMES 0-based index of the frame to be decoded/captured next.
# 2. CV_CAP_PROP_POS_AVI_RATIO Relative position of the video file
# 3. CV_CAP_PROP_FRAME_WIDTH Width of the frames in the video stream.
# 4. CV_CAP_PROP_FRAME_HEIGHT Height of the frames in the video stream.
# 5. CV_CAP_PROP_FPS Frame rate.
# 6. CV_CAP_PROP_FOURCC 4-character code of codec.
# 7. CV_CAP_PROP_FRAME_COUNT Number of frames in the video file.
# 8. CV_CAP_PROP_FORMAT Format of the Mat objects returned by retrieve() .
# 9. CV_CAP_PROP_MODE Backend-specific value indicating the current capture mode.
# 10. CV_CAP_PROP_BRIGHTNESS Brightness of the image (only for cameras).
# 11. CV_CAP_PROP_CONTRAST Contrast of the image (only for cameras).
# 12. CV_CAP_PROP_SATURATION Saturation of the image (only for cameras).
# 13. CV_CAP_PROP_HUE Hue of the image (only for cameras).
# 14. CV_CAP_PROP_GAIN Gain of the image (only for cameras).
# 15. CV_CAP_PROP_EXPOSURE Exposure (only for cameras).
# 16. CV_CAP_PROP_CONVERT_RGB Boolean flags indicating whether images should be converted to RGB.
# 17. CV_CAP_PROP_WHITE_BALANCE Currently unsupported
# 18. CV_CAP_PROP_RECTIFICATION Rectification flag for stereo cameras (note: only supported by DC1394 v 2.x backend currently)



#initialize networkTables
logging.basicConfig(level=logging.DEBUG)
NetworkTables.initialize()
sd = NetworkTables.getTable("SmartDashboard")
coords = NetworkTables.getTable("coordinates")
res = NetworkTables.getTable("Resolution")

while(True):

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    ret, frame = video.read()
    frame_expanded = np.expand_dims(frame, axis=0)
    height, width, channels = frame.shape

    #inputs camera resolution to networktables
    res.putNumber("Height", height)
    res.putNumber("Width", width)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    # Draw the results of the detection (aka 'visulaize the results')
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.60)

    
    # print("dsTime:", hf.getNumber("robotTime", i))

    # hf.putNumber("robotTime", i)
    # time.sleep(0)
    # i = i+1
    numDetections = [category_index.get(value) for index,value in enumerate(classes[0]) if scores[0,index] > 0.5]

    ymin = []
    xmin = []
    ymax = []
    xmax = []
    widthOfBox = []
    heightOfBox = []
    centerXCoordinates = []
    centerYCoordinates = []

                
    # top left is: xmin, ymin
    # bottom right is xmax, ymax
    #centerXCoordinates formula (xmin + xmax) / 2, (ymin + ymax) / 2
                
    flag = 0  
    for x in numDetections:
        yminVal = int((boxes[0][flag][0]*height))
        xminVal = int((boxes[0][flag][1]*width))
        ymaxVal = int((boxes[0][flag][2]*height))
        xmaxVal = int((boxes[0][flag][3]*width))

        ymin.append(yminVal)
        xmin.append(xminVal)
        ymax.append(ymaxVal)
        xmax.append(xmaxVal)

        widthOfBox.append(xmax[flag] - xmin[flag])
        heightOfBox.append(ymax[flag] - ymin[flag])

        centerXCoordinates.append((xmin[flag] + xmax[flag]) / 2)
        centerYCoordinates.append((ymin[flag] + ymax[flag]) / 2)

        coords.putNumberArray("ymin", ymin)
        coords.putNumberArray("ymax", ymax)
        
        coords.putNumberArray("xmin", xmin)
        coords.putNumberArray("xmax", xmax)
        
        coords.putNumberArray("centerX", centerXCoordinates)
        coords.putNumberArray("centerY", centerXCoordinates)

        coords.putNumberArray("boxWidth", widthOfBox)
        coords.putNumberArray("boxHeight", heightOfBox)
        

        cv2.circle(frame,(xminVal, yminVal), 10, (0,0,255), -1)
        cv2.circle(frame,(xmaxVal, ymaxVal), 10, (0,0,255), -1)
        cv2.circle(frame,(xmaxVal, yminVal), 10, (0,0,255), -1)
        cv2.circle(frame,(xminVal, ymaxVal), 10, (0,0,255), -1)
        cv2.circle(frame,(int(centerXCoordinates[flag]), int(centerYCoordinates[flag])), 20, (0,0,255), -1)
        flag += 1

    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
video.release()
cv2.destroyAllWindows()

