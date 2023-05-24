import math
import random

import cv2
import threading
import os
import time
import numpy as np
from ultralytics import YOLO

#
# script sends frames to the display thread as soon as they are available.
# buffering is done
# the frame time calculation is done in the broadcast thread
# the display thread lags if the processing time is more than the frame delay
# video speed is preserved
#


# Definirajte globalno spremenljivko za shranjevanje izbrane funkcije predprocesiranja
selected_preprocessing = None
opt1 = None
opt2 = None
opt3 = None
opt4 = None

# Nit za določanje funkcije predprocesiranja
class PreprocessingThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.stopped = False

    def run(self):
        global selected_preprocessing
        global opt1
        global opt2
        global opt3
        global opt4

        # Neskončna zanka za branje uporabnikovih vhodov
        while not self.stopped:
            # Preberite uporabnikov vhod
            user_input = input("Vnesite besedo za določanje funkcije predprocesiranja (ali 'q' za izhod): ")
            # Preverite, ali uporabnik želi izhod
            if user_input == "q":
                break

            # Preverite uporabnikov vhod in nastavite izbrano funkcijo predprocesiranja
            # Parsanje uporabnikovega vhoda
            array = user_input.split(" ")
            if len(array) > 5:
                print("Neveljaven vhod. Prosimo, poskusite znova.")
            elif len(array) == 5:
                selected_preprocessing, opt1, opt2, opt3, opt4 = array
            elif len(array) == 4:
                selected_preprocessing, opt1, opt2, opt3 = array
            elif len(array) == 3:
                selected_preprocessing, opt1, opt2 = array
            elif len(array) == 2:
                selected_preprocessing, opt1 = array
            elif len(array) == 1:
                selected_preprocessing = array[0]
            else:
                print("Neveljaven vhod. Prosimo, poskusite znova.")

    def stop(self):
        self.stopped = True


class VideoBroadcastThread(threading.Thread):
    def __init__(self, video_path, buffer_size):
        threading.Thread.__init__(self)
        self.video_path = video_path
        self.buffer_size = buffer_size
        self.frame_buffers = [None, None]
        self.current_buffer = 0
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

        while not self.stopped:
            while cap.isOpened() and not self.stopped:
                start_time = time.time()

                # Read frame from the video
                ret, frame = cap.read()
                if not ret:
                    break

                # Add frame to the current buffer
                self.frame_buffers[self.current_buffer] = frame

                # Switch to the other buffer if it's not None
                self.current_buffer = 1 - self.current_buffer
                if self.frame_buffers[self.current_buffer] is not None:
                    # Wait for the receiving thread to consume frames
                    time.sleep(frame_delay)

                elapsed_time = time.time() - start_time
                delay = max(0, frame_delay - elapsed_time)
                time.sleep(delay)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

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
            # Check if the current buffer has a frame available
            frame = self.broadcast_thread.frame_buffers[self.broadcast_thread.current_buffer]
            if frame is not None:
                # Display the frame
                frame_count += 1

                if selected_preprocessing is not None:
                    # Uporabite izbrano funkcijo predprocesiranja
                    frame = process_frame(frame)

                cv2.imshow('Video Display', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # Set the frame in the current buffer to None
                self.broadcast_thread.frame_buffers[self.broadcast_thread.current_buffer] = None
            else:
                # If the current buffer is empty, wait for a frame to be available
                time.sleep(0.001)

        # Close windows
        cv2.destroyAllWindows()

        # Signal the end of the display
        self.stopped = True

    def stop(self):
        self.stopped = True

def process_frame(frame):
    global selected_preprocessing
    global opt1
    global opt2
    global opt3
    global opt4

    if selected_preprocessing == "add_noise":
        return cv2.add(frame, np.random.normal(0, 1, frame.shape).astype(np.uint8))
    elif selected_preprocessing == "resize":
        output_size = (int(opt1), int(opt2)) # (9, 9)
        kernel_size = (3, 3)
        sigma = 1.5
        return cv2.resize(cv2.GaussianBlur(frame, kernel_size, sigma), output_size, interpolation=cv2.INTER_AREA)
    elif selected_preprocessing == "add_background":
        return add_background(frame, opt1)
    elif selected_preprocessing == "high-pass":
        kernel_size = (int(opt1), int(opt2))
        return cv2.subtract(frame, cv2.blur(frame, kernel_size))
    elif selected_preprocessing == "low-pass":
        kernel_size = int(opt1)
        return cv2.blur(frame, kernel_size)
    elif selected_preprocessing == "frequency_shift":
        shift = (int(opt1), int(opt2))
        return shift_in_frequency_domain(frame, shift)
    elif selected_preprocessing == "rotate_clockwise":
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif selected_preprocessing == "rotate_counter_clockwise":
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        # Neznani ID procesa, vrnemo nespremenjen frejm
        return frame

def add_background(frame, background_image_path):
    # Implementacija dodajanja posnetkov ozadja
    # Tukaj je primer, kako lahko dodate posnetek ozadja na frejm:
    background_image = cv2.imread(background_image_path)
    background_image = cv2.resize(background_image, (frame.shape[1], frame.shape[0]))
    background_frame = cv2.addWeighted(frame, 0.8, background_image, 0.2, 0)
    return background_frame

def shift_in_frequency_domain(frame, shift):
    # Preveri, če je število vrstic in stolpcev sodo število
    rows, cols, _ = frame.shape
    if rows % 2 != 0 or cols % 2 != 0:
        raise ValueError("The number of rows and columns of the frame must be even.")

    # Pretvorba v sivinsko sliko
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Izvedba diskretne Fourierjeve transformacije (DFT)
    dft = np.fft.fft2(gray_frame)

    # Premik frekvenc v frekvenčnem prostoru
    shifted_dft = np.fft.fftshift(dft)

    # Izvedba inverzne DFT
    shifted_frame = np.fft.ifft2(shifted_dft)

    # Pretvorba nazaj v barvni prostor
    shifted_frame = cv2.cvtColor(np.real(shifted_frame), cv2.COLOR_GRAY2BGR)

    return shifted_frame


# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Main program
if __name__ == '__main__':
    folder_path = 'no_audio'
    video_name = 'video_854x480.mp4'
    video_path = os.path.join(folder_path, video_name)

    # Set detection video properties
    detection_width = 854
    detection_height = 480

    # Set buffer size and create and start the threads
    buffer_size = 10  # Adjust the buffer size as per your requirements
    broadcast_thread = VideoBroadcastThread(video_path, buffer_size)
    display_thread = VideoDisplayThread(broadcast_thread)
    preprocessing_thread = PreprocessingThread()


    broadcast_thread.start()
    time.sleep(0.5)  # Wait for the broadcast thread to start and initialize the video properties
    display_thread.start()
    preprocessing_thread.start()

    # Wait for the display thread to finish or the video to end
    while not display_thread.stopped and not broadcast_thread.stopped and not preprocessing_thread.stopped:
        time.sleep(0.1)

    # Stop the threads
    display_thread.stop()
    broadcast_thread.stop()
    preprocessing_thread.stop()

    # Wait for the threads to finish
    display_thread.join()
    broadcast_thread.join()
    preprocessing_thread.join()
