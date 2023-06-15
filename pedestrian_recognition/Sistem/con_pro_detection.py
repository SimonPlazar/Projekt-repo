import json
import math

import numpy as np
import cv2
import threading
import redis
import kafka
import signal
from datetime import datetime
import time
import torch
from kafka.structs import TopicPartition
from prometheus_client import start_http_server, Counter, Enum, Gauge
from ultralytics import YOLO

# Prometheus metrics
# More:https://github.com/prometheus/client_python
counter_neuralnetwork = Counter('nn_detections', 'Number of NN detections')
counter_detections_per_frame= Counter('detections_per_frame', 'Number of detections per frame')
detection_state = Enum('detection_state', 'State of the detection', states=['starting', 'running', 'stopped'])
gauge_detection_size = Counter('detection_size', 'Size of the detected object')

def thread_do_work():

    # Redis
    red = redis.Redis()

    # Kafka
    con_topic = 'frame_noticifation'
    prod_topic = 'frame_detection'
    producer = kafka.KafkaProducer(bootstrap_servers='localhost:9092')

    # auto_offet_reset indicates where the consumer starts consuming the message in case of a interruption
    consumer = kafka.KafkaConsumer(bootstrap_servers='localhost:9092', auto_offset_reset='earliest', group_id='grp_detection', consumer_timeout_ms=2000)
    consumer.subscribe([con_topic])

    # PyTorch
    model = YOLO("yolov8n.pt")

    while True:
        detection_state.state('running')
        # Read from Redis when message is received over Kafka
        for message in consumer:
            if message.value.decode("utf-8") == "new_frame":
                frame_temp = np.frombuffer(red.get("frame:latest"), dtype=np.uint8)
                width = None
                height = None

                # Convert image
                if (np.shape(frame_temp)[0] is not None):
                    # Extract the header value
                    header = message.headers
                    for header_key, header_value in header:
                        if header_key == 'video_info':
                            # Extract video width and height from the header
                            header_json = header_value.decode('utf-8')
                            header = json.loads(header_json)
                            width = header['video_width']
                            height = header['video_height']

                    # Reshape the frame
                    if width is not None and height is not None:
                        frame = frame_temp.reshape((height, width, 3))
                    else:
                        print("Width and height are not set")
                        if(np.shape(frame_temp)[0] == 6220800):
                            frame = frame_temp.reshape((480, 854, 3))
                        elif(np.shape(frame_temp)[0] == 2764800):
                            frame = frame_temp.reshape((720, 1280, 3))
                        elif(np.shape(frame_temp)[0] == 1229760):
                            frame = frame_temp.reshape((480, 854, 3))

                    # Detection
                    results = model.predict(frame, conf=0.59, classes=[0], verbose=False)

                    preds_list = []

                    for boxes in [param.boxes.data for param in results[0].numpy()]:
                        for box in boxes:
                            preds_list.append(','.join(str(round(element)) for element in box[:4]))
                            box_size = math.dist((box[0], box[1]), (box[2], box[3]))
                            gauge_detection_size.inc(box_size)

                    counter_detections_per_frame.inc(preds_list.__len__())

                    # Send detection data over Kafka
                    # TODO: Not the best way of sending and reading data with Kafka
                    #       Modify sending and receiving using value_serializer and value_deserializer and work with JSON on both sides
                    future = producer.send(prod_topic, str.encode("|".join(preds_list)), timestamp_ms=round(time.time()*1000))

                    # Wait until message is delivered to Kafka
                    try:
                        rm = future.get(timeout=10)
                    except kafka.KafkaError:
                        pass 

                    # Prometheus metrics
                    counter_neuralnetwork.inc(1) 
            
            time.sleep(1)

            # To beginning
            consumer.seek_to_beginning()
            consumer.commit()

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
    # Prometheus metrics
    start_http_server(8000)

    detection_state.state('starting')

    thread.start()
    input("Press CTRL+C or Enter to stop visualization...")
    event.set()
    thread.join()

    detection_state.state('stopped')