import os
import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Imput path
video_name = "meritev_1682262004_002.mp4"
video_folder = "no_audio"
video_path = os.path.join(video_folder, video_name)

# Load video
video = cv2.VideoCapture(video_path)

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
output_width = width
output_height = height

# Output path
output_video_name = "output_" + video_name
output_video_folder = "Output"
output_video_path = os.path.join(output_video_folder, output_video_name)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, output_height))

# Initialize list for persisting bounding boxes
prev_bounding_boxes = np.array([])
new_bounding_boxess = np.array([])
frame_count = 0

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
        cv2.putText(frame, "Person " + str(round(box[4], 3)), (int(box[0]*width_factor), int(box[1]*height_factor) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.rectangle(frame, (int(box[0]*width_factor), int(box[1]*height_factor)), (int(box[2]*width_factor), int(box[3]*height_factor)), (0, 255, 0), 2)

    # Write frame to output video
    output_video.write(frame)
    frame_count += 1

# Release resources
video.release()
output_video.release()
cv2.destroyAllWindows()
