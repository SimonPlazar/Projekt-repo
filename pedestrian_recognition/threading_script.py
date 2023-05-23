import cv2
import threading
import os

class VideoBroadcastThread(threading.Thread):
    def __init__(self, video_path):
        threading.Thread.__init__(self)
        self.video_path = video_path
        self.frame_available = threading.Event()
        self.frame = None
        self.stopped = False

    def run(self):
        # Open video capture
        cap = cv2.VideoCapture(self.video_path)

        while cap.isOpened() and not self.stopped:
            # Read frame from the video
            ret, frame = cap.read()
            if not ret:
                break

            # Set the frame in shared variable
            self.frame = frame
            self.frame_available.set()
            self.frame_available.clear()

        # Release video capture
        cap.release()

    def stop(self):
        self.stopped = True

class VideoDisplayThread(threading.Thread):
    def __init__(self, broadcast_thread):
        threading.Thread.__init__(self)
        self.broadcast_thread = broadcast_thread
        self.stopped = False

    def run(self):
        while not self.stopped:
            # Wait for a frame to be available
            self.broadcast_thread.frame_available.wait()

            # Get the frame from the broadcast thread
            frame = self.broadcast_thread.frame

            # Display the frame
            cv2.imshow('Video Display', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Close windows
        cv2.destroyAllWindows()

    def stop(self):
        self.stopped = True

# Main program
if __name__ == '__main__':
    folder_path = 'no_audio'
    video_name = 'video_1280x720.mp4'
    video_path = os.path.join(folder_path, video_name)

    # Create and start the threads
    broadcast_thread = VideoBroadcastThread(video_path)
    display_thread = VideoDisplayThread(broadcast_thread)

    broadcast_thread.start()
    display_thread.start()

    # Wait for the user to terminate the program
    input("Press Enter to stop...")

    # Stop the threads
    display_thread.stop()
    broadcast_thread.stop()

    # Wait for the threads to finish
    display_thread.join()
    broadcast_thread.join()

