import math
import cv2
import threading
import os
import time
import numpy as np
from ultralytics import YOLO

#
# script sends frames to the display thread as soon as they are available.
# no buffering is done
# the frame time calculation is done in the broadcast thread
#

class VideoBroadcastThread(threading.Thread):
    def __init__(self, video_path):
        threading.Thread.__init__(self)
        self.video_path = video_path
        self.frame_available = threading.Event()
        self.frame = None
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

            # Set the frame in shared variable
            self.frame = frame
            self.frame_available.set()
            self.frame_available.clear()

            elapsed_time = time.time() - start_time
            delay = max(0, frame_delay - elapsed_time)
            time.sleep(delay)

        # Release video capture
        cap.release()

        # Signal the end of the video
        self.frame_available.set()

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
            # Wait for a frame to be available
            self.broadcast_thread.frame_available.wait()

            # Get the frame from the broadcast thread
            frame = self.broadcast_thread.frame

            # Process the frame
            if frame is not None:
                # Display the frame
                frame_count += 1

                # simulate processing time
                time.sleep(0.1)

                cv2.imshow('Video Display', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        # Close windows
        cv2.destroyAllWindows()

        # Signal the end of the display
        self.stopped = True

    def stop(self):
        self.stopped = True

# Load YOLOv8 model
model = YOLO("../yolov8n.pt")

# Main program
if __name__ == '__main__':
    folder_path = '../no_audio'
    video_name = 'video_1280x720.mp4'
    video_path = os.path.join(folder_path, video_name)

    # Set detection video properties
    detection_width = 854
    detection_height = 480

    # Create and start the threads
    broadcast_thread = VideoBroadcastThread(video_path)
    display_thread = VideoDisplayThread(broadcast_thread)

    broadcast_thread.start()
    time.sleep(0.1) # Wait for the broadcast thread to start and initialize the video properties
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
