import numpy as np
import threading
import redis
import kafka
import signal
import cv2 
from datetime import datetime
import time
import sys
sys.path.append('C:/Users/PC/Desktop/Projekt/')
from functions import preprocess_image


def thread_do_work():
    red = redis.Redis()
    topic = 'frame_noticifation'
    producer = kafka.KafkaProducer(bootstrap_servers='localhost:9092')
    fps = 30
    input = "../output1.mp4"
    cap = cv2.VideoCapture(input) #cv2.CAP_DSHOW
    #ret, frame = cap.read()
    #cv2.imshow("Video", frame)
    while True:
        t_start = time.perf_counter()
        ret, frame = cap.read()
        
        if not ret:
           cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        #print(frame.shape)
        red.set("frame:latest", np.array(frame).tobytes())

        #cv2.imshow("Video", frame)
        #cv2.waitKey(1)

        future = producer.send(topic, b"new_frame", timestamp_ms=round(time.time()*1000))

        try:
            rm = future.get(timeout=10)
        except kafka.KafkaError:
            pass
        


        t_stop = time.perf_counter()
        t_elapsed = t_stop - t_start
        t_frame = 1000 / fps / 1000
        t_sleep = t_frame - t_elapsed
        if t_sleep > 0:
            time.sleep(t_sleep)

        if event.is_set():
            cap.release()
            #cv2.destroyAllWindows()
            break 


def sigint_handler(signum, frame):
    event.set()
    thread.join()
    exit(0)

signal.signal(signal.SIGINT, sigint_handler)
event = threading.Event()
thread = threading.Thread(target=lambda: thread_do_work())

if __name__ == "__main__":
    thread.start()
    input("Press CTRL+C or Enter to stop visualization...")
    event.set()
    thread.join()
