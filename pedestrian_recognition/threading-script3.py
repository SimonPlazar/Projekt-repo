import math
import cv2
import threading
import os
import time
import numpy as np
from ultralytics import YOLO

#
# VERY SLOW WITH PROCESSING DELAY
#

class VideoBroadcastThread(threading.Thread):
    def __init__(self, video_path, buffer_size):
        threading.Thread.__init__(self)
        self.video_path = video_path
        self.buffer_size = buffer_size
        self.frame_buffers = []
        self.stopped = False
        self.width = None
        self.height = None
        self.fps = None

    def run(self):
        # Open video capture
        cap = cv2.VideoCapture(self.video_path)

        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_delay = 1 / self.fps

        while cap.isOpened() and not self.stopped:
            start_time = time.time()

            # Read frame from the video
            ret, frame = cap.read()
            if not ret:
                break

            # Add frame to the buffer
            self.frame_buffers.append(frame)

            # Wait for the buffer to have available space
            while len(self.frame_buffers) >= self.buffer_size:
                time.sleep(0.001)

            elapsed_time = time.time() - start_time
            delay = max(0, frame_delay - elapsed_time)
            time.sleep(delay)

        # Release video capture
        cap.release()

        # Signal the end of the video
        self.stopped = True

    def stop(self):
        self.stopped = True

class VideoDisplayThread(threading.Thread):
    def __init__(self, broadcast_thread):
        threading.Thread.__init__(self)
        self.broadcast_thread = broadcast_thread
        self.stopped = False

    def run(self):
        # Factor for resizing bounding boxes
        width_factor = self.broadcast_thread.width / detection_width
        height_factor = self.broadcast_thread.height / detection_height
        frame_count = 0

        while not self.stopped:
            # Check if the buffer has a frame available
            if len(self.broadcast_thread.frame_buffers) > 0:
                # Get the frame from the buffer
                frame = self.broadcast_thread.frame_buffers.pop(0)

                # Process the frame (e.g., post-processing)
                processed_frame = process_frame(frame)

                # Display the frame
                frame_count += 1
                cv2.imshow('Video Display', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                # If the buffer is empty, wait for a frame to be available
                time.sleep(0.001)

        # Close windows
        cv2.destroyAllWindows()

        # Signal the end of the display
        self.stopped = True

    def stop(self):
        self.stopped = True

def process_frame(frame):
    # Perform post-processing operations on the frame
    # Example: Apply filters, resize, object detection, etc.
    processed_frame = frame  # Placeholder, replace with actual processing code
    time.sleep(0.1)  # Simulate processing time
    return processed_frame

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Main program
if __name__ == '__main__':
    folder_path = 'no_audio'
    video_name = 'video_1280x720.mp4'
    video_path = os.path.join(folder_path, video_name)

    # Set detection video properties
    detection_width = 854
    detection_height = 480

    # Set buffer size and create and start the threads
    buffer_size = 10  # Adjust the buffer size as per your requirements
    broadcast_thread = VideoBroadcastThread(video_path, buffer_size)
    display_thread = VideoDisplayThread(broadcast_thread)

    broadcast_thread.start()
    time.sleep(0.1)  # Wait for the broadcast thread to start and initialize the video properties
    display_thread.start()

    # Wait for the display thread to finish or the video to end
    while not display_thread.stopped and not broadcast_thread.stopped:
        time.sleep(0.1)

    # Stop the threads
    display_thread.stop()
    broadcast_thread.stop()

    # Wait for the threads to finish
    display_thread.join()
    broadcast_thread.join()
