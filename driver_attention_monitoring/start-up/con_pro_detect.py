import numpy as np
import cv2 as cv
import threading
import redis
import kafka
import signal
from datetime import datetime
import time
from kafka.structs import TopicPartition
from prometheus_client import start_http_server, Counter, Gauge, Enum
import sys
import torch
sys.path.append('C:/Users/PC/Desktop/Projekt/')
from functions import load_model_by_weights, preprocess_image
from setproctitle import setproctitle
import psutil
import os

#setproctitle("con_pro_detect.py")

counter_oci = Counter('nn_eye_detections', 'Number of eye detections')
counter_oci_zaprte = Counter('nn_eye_detections_zaprte', 'zaprte')
counter_obraz = Counter('nn_head_detections', 'Number of head detections')
counter_obraz_utrujen = Counter('nn_head_detections_utrujen', 'utrujen')
cpu_p = Gauge('cpu_process', 'CPU process')
mem_p = Gauge('mem_process', 'Memory process')
e_oci = Enum('stanje_oci', 'stanje v katerem so oci', states=['odprte', 'zaprte'])
e_obraz = Enum('stanje_obraz', 'stanje v katerem je obraz', states=['ne_utrujen', 'utrujen'])


color_map = {
    "ne_utrujen" : (0, 255, 0), #zelena
    "utrujen" : (0, 0, 255), #rdeca

    "zaprte" : (0, 0, 255), #rdeca
    "odprte" : (0, 255, 0) #zelena
}

labels_oci = ["odprte"]
labels_utrujenost = ["ne_utrujen"]

def detect(results, frame):
    global labels_oci, labels_utrujenost

    #vzame vse x1,y1,x2,y2 kordinate iz detektiranih boxov
    boxes = results.xyxy[0].numpy()[:, :4]

    #ce ni framov konca
    if len(boxes) > 0:

        #izracuna povrsino boxov
        areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        #najde najvecjo povrsino
        max_area_index = np.argmax(areas)
        najvecji = results.xyxy[0][max_area_index]

        #vzame kordinate najvecjega boxa
        x1, y1, x2, y2 = map(int,najvecji[:4])

        #in label
        label = results.names[int(najvecji[-1])]

        #conf = float(najvecji[-2])

        color_box = color_map[label]

        #narise box in napise label
        cv.rectangle(frame, (x1, y1), (x2, y2), color_box, 1)
        #cv.putText(frame, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        return label
    
    return None


def thread_do_work():
    global labels_oci, labels_utrujenost
    red = redis.Redis()
    pid = os.getpid()
    p = psutil.Process(pid)

    con_topic = 'frame_noticifation'
    prod_topic = 'frame_detection'
    prod_topic2 = 'frame_rezultati'
    producer = kafka.KafkaProducer(bootstrap_servers='localhost:9092')

    consumer = kafka.KafkaConsumer(bootstrap_servers='localhost:9092', auto_offset_reset='earliest', group_id='grp_detection', consumer_timeout_ms=2000)
    consumer.subscribe([con_topic])

    model_oci = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/PC/Desktop/Projekt/yolov5obraz/runs/train/exp5/weights/last.pt') #, force_reload=True
    model_utrujenost = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/PC/Desktop/Projekt/yolov5/runs/train/exp2/weights/last.pt') #, force_reload=True

    while True:
    
        # Read from Redis when message is received over Kafka
        for message in consumer:

            if message.value.decode("utf-8") == "new_frame":
                frame_time = datetime.fromtimestamp(message.timestamp / 1000)
                curr_time = datetime.now()
                diff = (curr_time - frame_time).total_seconds() 
                
                # Exclude old frames
                if diff < 2:
                    frame_temp = np.frombuffer(red.get("frame:latest"), dtype=np.uint8)
                    
                #     # Convert image
                    if (np.shape(frame_temp)[0] == 921600):
                        frame_temp = frame_temp.reshape((480, 640, 3))

                        results_oci = model_oci(frame_temp)
                        results_utrujenost = model_utrujenost(frame_temp)

                        label_oci = detect(results_oci, frame_temp)
                        if label_oci is not None:
                            labels_oci = label_oci
                            #print(label_oci)
                            #print(labels_oci.type())
                            if label_oci == "ne_utrujen":
                                #print("odprte")
                                e_obraz.state('ne_utrujen')
                                #g_oci.set(1)
                            else:
                                e_obraz.state('utrujen')
                                counter_obraz_utrujen.inc()


                            counter_oci.inc()
                        
                        label_utrujenost = detect(results_utrujenost, frame_temp)
                        if label_utrujenost is not None:
                            labels_utrujenost = label_utrujenost
                            #print(label_utrujenost)
                            #print(labels_utrujenost)
                            if label_utrujenost == "odprte":
                                e_oci.state('odprte')
                            else:
                                e_oci.state('zaprte')
                                counter_oci_zaprte.inc()

                            counter_obraz.inc()


                        cv.imshow("frame", frame_temp)
                        cv.waitKey(1)

                        send = labels_oci + "," + labels_utrujenost
                        future = producer.send(prod_topic2, str.encode(send), timestamp_ms=round(time.time()*1000))

                        try:
                            rm = future.get(timeout=10)
                        except kafka.KafkaError:
                            pass

                        cpu_usage = p.cpu_percent(interval=0.1)
                        mem_usage = p.memory_percent()
                        #print(cpu_usage)
                        cpu_p.set(cpu_usage / 10)
                        mem_p.set(mem_usage)

            if event.is_set():
                break   

        # Stop loop
        if event.is_set():
            break  


def sigint_handler(signum, frame):
    event.set()
    thread.join()
    exit(0)

signal.signal(signal.SIGINT, sigint_handler)

event = threading.Event()
thread = threading.Thread(target=lambda: thread_do_work())

if __name__ == "__main__":
    start_http_server(8000)
    
    thread.start()
    input("Press CTRL+C or Enter to stop visualization...")
    event.set()
    thread.join()