import numpy as np
import cv2
import threading
import redis
import kafka
import signal
from datetime import datetime

preds_list = []
preds_time = 0
lock = threading.Lock()

def thread_detection():
    global preds_list, preds_time

    # Kafka
    topic = 'frame_detection'
    consumer = kafka.KafkaConsumer(bootstrap_servers='localhost:9092', auto_offset_reset='earliest', consumer_timeout_ms=2000)
    consumer.subscribe([topic])

    while True:
    
        # Read detection data from Kafka
        for message in consumer:

            # Decode string
            preds_str = message.value.decode("utf-8")

            with lock:
                preds_list = preds_str.split("|") if len(preds_str) > 0 else []
                preds_time = datetime.fromtimestamp(message.timestamp / 1000)

            if event.is_set():
                break   

        # Stop loop
        if event.is_set():
            cv2.destroyAllWindows()
            break     

def thread_frames():
    global preds_list, preds_time

    # Redis
    red = redis.Redis()

    # Video
    frame = 0

    # Kafka
    topic = 'frame_noticifation'
    topic2 = 'frame_rezultati'
    consumer = kafka.KafkaConsumer(bootstrap_servers='localhost:9092', auto_offset_reset='earliest', group_id='grp_visualization', consumer_timeout_ms=2000)
    consumer.subscribe([topic])
    # consumer2 = kafka.KafkaConsumer(bootstrap_servers='localhost:9092', auto_offset_reset='earliest', consumer_timeout_ms=2000)
    # consumer2.subscribe([topic2])
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
                    
                    # Convert image
                    if (np.shape(frame_temp)[0] == 921600):
                        frame_temp = frame_temp.reshape((480, 640, 3))

                    if (curr_time - preds_time).total_seconds() < 5:
                        with lock:
                            for pred_str in preds_list:
                                # column_start,row_start,column_end,row_end
                                pred = [int(float(v)) for v in pred_str.split(",")]
                                # if len(pred) == 1:
                                #     break
                                # Top
                                cv2.line(frame_temp, (pred[0], pred[1]), (pred[2], pred[1]), (0, 0, 255), 5)
                                # Bottom
                                cv2.line(frame_temp, (pred[0], pred[3]), (pred[2], pred[3]), (0, 0, 255), 5)
                                # Left
                                cv2.line(frame_temp, (pred[0], pred[1]), (pred[0], pred[3]), (0, 0, 255), 5)
                                # Right
                                cv2.line(frame_temp, (pred[2], pred[1]), (pred[2], pred[3]), (0, 0, 255), 5)

                                if len(pred) == 5:
                                    cv2.putText(frame, f"Nagnjenost glave: {pred[4]}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Display image
                        cv2.imshow("frame", frame_temp)
                        cv2.waitKey(1)

            if event.is_set():
                break   

        # Stop loop
        if event.is_set():
            cv2.destroyAllWindows()
            break    

def sigint_handler(signum, frame):
    event.set()
    thread_frm.join()
    thread_det.join()
    exit(0)

signal.signal(signal.SIGINT, sigint_handler)

event = threading.Event()
thread_frm = threading.Thread(target=lambda: thread_frames())
thread_det = threading.Thread(target=lambda: thread_detection())

if __name__ == "__main__":
    thread_frm.start()
    thread_det.start()
    input("Press CTRL+C or Enter to stop visualization...")
    event.set()
    thread_frm.join()
    thread_det.join()