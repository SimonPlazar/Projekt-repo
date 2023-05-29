import cv2
from ultralytics import YOLO
from pytictoc import TicToc
import math

def Detect(video_path, output_video_path):

    # Load YOLOv8 model
    model = YOLO("yolov8n.pt")

    # Load video
    video = cv2.VideoCapture(video_path)

    # Get video properties
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    # Set detection video properties
    detection_width = 1280
    detection_height = 720

    # Factor for resizing bounding boxes
    width_factor = width / detection_width
    height_factor = height / detection_height

    # Set output video properties
    output_width = width
    output_height = height

    # Output path
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, output_height))

    # Initialize list for persisting bounding boxes
    frame_count = 0

    t = TicToc()
    t.tic()
    while True:
        # Read frame
        ret, frame = video.read()

        if not ret:
            break

        frame_small = cv2.resize(frame, (detection_width, detection_height))
        results = model.predict(frame_small, conf=0.59, classes=[0], verbose=False)[0].numpy()

        # Draw each box from results directly
        for param in results:
            for box in param.boxes.data:
                box_size = math.dist((box[0], box[1]), (box[2], box[3]))
                cv2.putText(frame, "Person " + str(round(box[4], 3)),
                    (int(box[0] * width_factor), int(box[1] * height_factor) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                if box_size > 180:
                    cv2.rectangle(frame, (int(box[0] * width_factor), int(box[1] * height_factor)),
                        (int(box[2] * width_factor), int(box[3] * height_factor)), (0, 0, 255), 2)
                elif box_size > 100:
                    cv2.rectangle(frame, (int(box[0] * width_factor), int(box[1] * height_factor)),
                        (int(box[2] * width_factor), int(box[3] * height_factor)), (0, 165, 255), 2)
                else:
                    cv2.rectangle(frame, (int(box[0] * width_factor), int(box[1] * height_factor)),
                        (int(box[2] * width_factor), int(box[3] * height_factor)), (0, 255, 0), 2)
        # Write frame to output video
        output_video.write(frame)
        frame_count += 1
    t.toc()

    # Release resources
    video.release()
    output_video.release()
    cv2.destroyAllWindows()
