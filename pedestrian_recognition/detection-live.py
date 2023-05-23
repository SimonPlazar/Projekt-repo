import os
import cv2
import numpy as np
from ultralytics import YOLO
from pytictoc import TicToc
import math

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Load video
video = cv2.VideoCapture(0)

# Get video properties
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))

# Set detection video properties
detection_width = 854
detection_height = 480

# Factor for resizing bounding boxes
width_factor = width / detection_width
height_factor = height / detection_height

# Set output video properties
output_width = 1280
output_height = 720

# Initialize list for persisting bounding boxes
prev_bounding_boxes = np.array([])
new_bounding_boxess = np.array([])
frame_count = 0

t = TicToc()
t.tic()
while True:
    # Read frame
    ret, frame = video.read()

    if not ret:
        break

    frame_small = cv2.resize(frame, (detection_width, detection_height))

    # Detect every 5th frame
    if frame_count % 7 == 0:
        results = model.predict(frame_small, conf=0.59, classes=[0]) #, verbose=False)
        results = results[0].numpy()

        prev_bounding_boxes = np.array([])
        if len(results) != 0:
            # Concatenate all boxes data
            prev_bounding_boxes = np.concatenate([param.boxes.data for param in results])

    # Draw each box from prev_bounding_boxes
    for box in prev_bounding_boxes:
        box_size = math.dist((box[0], box[1]), (box[2], box[3]))
        cv2.putText(frame, "Person " + str(round(box[4], 3)), (int(box[0]*width_factor), int(box[1]*height_factor) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if(box_size > 180):
            cv2.rectangle(frame, (int(box[0]*width_factor), int(box[1]*height_factor)), (int(box[2]*width_factor), int(box[3]*height_factor)), (0, 0, 255), 2)
        elif (box_size > 100):
            cv2.rectangle(frame, (int(box[0]*width_factor), int(box[1]*height_factor)), (int(box[2]*width_factor), int(box[3]*height_factor)), (0, 165, 255), 2)
        else:
            cv2.rectangle(frame, (int(box[0]*width_factor), int(box[1]*height_factor)), (int(box[2]*width_factor), int(box[3]*height_factor)), (0, 255, 0), 2)

    # Write frame to output video
    frame_count += 1
    frame_output = cv2.resize(frame, (output_width, output_height))
    cv2.imshow('Frame', frame_output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
t.toc()

# Release resources
video.release()
cv2.destroyAllWindows()
