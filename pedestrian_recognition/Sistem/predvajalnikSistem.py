import math
import sys
import cv2
import threading
import os
from datetime import datetime
import kafka
import numpy as np
import time
from ultralytics import YOLO
import getopt
import redis
from prometheus_client import start_http_server, Counter

#
# script sends frames to the display thread as soon as they are available.
# buffering is done
# the frame time calculation is done in the broadcast thread
# the display thread lags if the processing time is more than the frame delay
# video speed is preserved
#

# Prometheus metrics
# More:https://github.com/prometheus/client_python
counter_neuralnetwork = Counter('nn_detections', 'Number of NN detections')

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
                self.stopped = True
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
                selected_preprocessing, opt1, opt2, opt3, opt4 = None, None, None, None, None

    def stop(self):
        self.stopped = True



class VideoBroadcastThread(threading.Thread):
    def __init__(self, video_path, buffer_size, x, y):
        threading.Thread.__init__(self)
        self.video_path = video_path
        self.buffer_size = buffer_size
        self.frame_buffers = [None, None]
        self.current_buffer = 0
        self.stopped = False
        self.width = x
        self.height = y
        self.fps = None
        self.resize = True

    def run(self):
        # Redis
        red = redis.Redis()

        # Kafka
        topic = 'frame_noticifation'
        producer = kafka.KafkaProducer(bootstrap_servers='localhost:9092')

        # Open video capture
        # cap = cv2.VideoCapture(self.video_path)
        cap = cv2.VideoCapture(video_path)

        resize_x, resize_y = True, True
        sizeX, sizeY = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        if self.width is None:
            resize_x = False
            self.width = sizeX
        else:
            if self.width == sizeX:
                resize_x = False

        if self.height is None:
            resize_y = False
            self.height = sizeY
        else:
            if self.height == sizeY:
                resize_y = False

        if not resize_x and not resize_y:
            self.resize = False

        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_delay = 1 / self.fps

        while cap.isOpened() and not self.stopped:
            if not self.stopped:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            start_time = time.time()

            # Read frame from the video
            ret, frame = cap.read()
            if not ret:
                break

            if self.resize:
                frame = cv2.resize(frame, (self.width, self.height))

            # Add frame to redis
            red.set("frame:latest", np.array(frame).tobytes())

            # Send notification about new frame over Kafka
            future = producer.send(topic, b"new_frame", timestamp_ms=round(time.time() * 1000))

            # Wait until message is delivered to Kafka
            try:
                rm = future.get(timeout=10)
            except kafka.KafkaError:
                pass

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
        # Redis
        red = redis.Redis()

        # Kafka
        con_topic = 'frame_noticifation'
        prod_topic = 'frame_detection'
        producer = kafka.KafkaProducer(bootstrap_servers='localhost:9092')

        # auto_offet_reset indicates where the consumer starts consuming the message in case of a interruption
        consumer = kafka.KafkaConsumer(bootstrap_servers='localhost:9092', auto_offset_reset='earliest',
                                       group_id='grp_detection', consumer_timeout_ms=2000)
        consumer.subscribe([con_topic])

        frame_count = 0

        while not self.stopped:

            # Read from Redis when message is received over Kafka
            for message in consumer:
                if message.value.decode("utf-8") == "new_frame":

                    frame_temp = np.frombuffer(red.get("frame:latest"), dtype=np.uint8)

                    # Convert image
                    if (frame_temp is not None):
                        frame = frame_temp.reshape((480, 854, 3))

                        # Detection
                        if frame is not None:
                            # Display the frame
                            frame_count += 1

                            start_time = datetime.now()

                            # Uporabite izbrano funkcijo predprocesiranja
                            frame = detect(frame, 854, 480)

                            time_elapsed = datetime.now() - start_time

                            cv2.putText(frame, 'Time elapsed {}'.format(time_elapsed), (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                            cv2.putText(frame,
                                        'Frame delay ' + str(int(time_elapsed.microseconds - 1 / broadcast_thread.fps)),
                                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                            cv2.imshow('Video Display', frame)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break

                            # Set the frame in the current buffer to None
                            self.broadcast_thread.frame_buffers[self.broadcast_thread.current_buffer] = None

                        # Prometheus metrics
                        counter_neuralnetwork.inc(1)

                time.sleep(1)

                # To beginning
                consumer.seek_to_beginning()
                consumer.commit()

        # Close windows
        cv2.destroyAllWindows()

        # Signal the end of the display
        self.stopped = True

    def stop(self):
        self.stopped = True

def detect(frame, process_width, process_height):
    width_factor = frame.shape[1] / process_width
    height_factor = frame.shape[0] / process_height

    frame_small = cv2.resize(frame, (process_width, process_height))
    results = model.predict(frame_small, conf=0.59, classes=[0], verbose=False)

    prev_bounding_boxes = np.array([])
    if len(results[0].numpy()) != 0:
        # Concatenate all boxes data
        prev_bounding_boxes = np.concatenate([param.boxes.data for param in results[0].numpy()])

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

    return frame


# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Main program
if __name__ == '__main__':

    argv = sys.argv[1:]

    # pass argument to the program if video input or camera input
    opts, args = getopt.getopt(argv, "hi:co:", ["input=", "output="])

    folder_path = 'no_audio'
    video_name = 'video_854x480.mp4'
    video_path = os.path.join(folder_path, video_name)
    # video_path = 0

    x, y = None, None

    for opt, arg in opts:
        if opt == '-h':
            print('predvajalnikSistem.py [-i <video_name>][-c]')
            sys.exit()
        elif opt in ("-i"):
            video_name = arg
            video_path = os.path.join('no_audio', video_name)
        elif opt in ("-c"):
            video_path = 0
        elif opt in ("-o"):
            rez = arg
            x, y = rez.split('x')

    buffer_size = 50

    if x is not None and y is None:
        broadcast_thread = VideoBroadcastThread(video_path, buffer_size, int(x), None)
    elif x is None and y is not None:
        broadcast_thread = VideoBroadcastThread(video_path, buffer_size, None, int(y))
    elif x is not None and y is not None:
        broadcast_thread = VideoBroadcastThread(video_path, buffer_size, int(x), int(y))
    else:
        broadcast_thread = VideoBroadcastThread(video_path, buffer_size, None, None)
        
    display_thread = VideoDisplayThread(broadcast_thread)
    preprocessing_thread = PreprocessingThread()

    broadcast_thread.start()
    time.sleep(0.5)  # Wait for the broadcast thread to start and initialize the video properties
    display_thread.start()
    preprocessing_thread.start()

    # Wait for the display thread to finish or the video to end
    while not display_thread.stopped and not broadcast_thread.stopped and not preprocessing_thread.stopped:
        time.sleep(0.01)

    # Stop the threads
    display_thread.stop()
    broadcast_thread.stop()
    preprocessing_thread.stop()

    # Wait for the threads to finish
    display_thread.join()
    broadcast_thread.join()
    preprocessing_thread.join()