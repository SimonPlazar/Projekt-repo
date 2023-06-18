import numpy as np
import threading
import redis
import kafka
import signal
import cv2 as cv
from datetime import datetime
import time
import sounddevice as sd

# def thread_sound():
#     while True:
#         sd.play("Old_phone-ringing-sound.mp3", 44100)
#         time.sleep(0.5)
#         sd.stop()
#         if sound_event.is_set():
#             break

# def thread_sound(sound_event):
#     while not event.is_set():
#         if not sound_event.is_set():
#             sd.play("Old_phone-ringing-sound.mp3", 44100)
#             time.sleep(0.5)
#             sd.stop()

def thread_do_work():
    #red = redis.Redis()
    topic = 'frame_rezultati'
    consumer = kafka.KafkaConsumer(bootstrap_servers='localhost:9092', auto_offset_reset='earliest', consumer_timeout_ms=2000)
    consumer.subscribe([topic])

    eye_closed_counter = 0
    eye_open_counter = 0
    angle_over_x_counter = 0
    sound = False
    sound_thread = None

    while True:
        for message in consumer:
            #print("dobil sporočilo")
            decoded_message = message.value.decode("utf-8")
            split_message = decoded_message.split(",")
            #angle = float(split_message[0])
            eye = split_message[0]
            face = split_message[1]
            print(eye, face)
            #print( eye)

            # if eye == "zaprte":
            #     eye_closed_counter += 1
            #     eye_open_counter = 0
            # if eye == "odprte":
            #     eye_open_counter += 1
            #     eye_closed_counter = 0

            # # if abs(angle) > 25:
            # #     angle_over_x_counter += 1
            # # else:
            # #     angle_over_x_counter = 0


            # if eye_closed_counter > 4 and sound == False:
            #     print("START sound")
            #     #sound_thread.start()
            #     #sound_event.clear()
            #     sound = True
            #     #
            
            # if eye_open_counter > 2 and sound == True:
            #     print("STOP sound")
            #     #sound_thread.join()
            #     #sound_event.set()
            #     sound = False

            # if angle_over_x_counter > 3:
            #     sd.play("warning-sound-6686.mp3", 44100)
            #     angle_over_x_counter = 0

                #
            # if (angle < 25 and angle > -25) or eye_open_counter > 5:
            #     #stop thread sound
            #     print("stop sound")
            #     #pass
                


            
            #print(message.value.decode("utf-8"))
            #if message.value.decode("utf-8") == "new_frame":
            #     frame_time = datetime.fromtimestamp(message.timestamp / 1000)
            #     curr_time = datetime.now()
            #     diff = (curr_time - frame_time).total_seconds()


            if event.is_set():
                break   

        # Stop loop
        if event.is_set():
            break  


def sigint_handler(signum, frame):
    event.set()
    thread.join()
    exit(0)

# def sigint_handler(signum, frame):
#     event.set()
#     sound_event.set()
#     thread.join()
#     sound_thread.join()
#     exit(0)

signal.signal(signal.SIGINT, sigint_handler)
event = threading.Event()
sound_event = threading.Event()
thread = threading.Thread(target=lambda: thread_do_work())
#sound_thread = threading.Thread(target=lambda: thread_sound())

# signal.signal(signal.SIGINT, sigint_handler)
# event = threading.Event()
# sound_event = threading.Event()
# thread = threading.Thread(target=thread_do_work, args=(event, sound_event))
# sound_thread = threading.Thread(target=thread_sound, args=(sound_event,))

if __name__ == "__main__":
    thread.start()
    input("Press CTRL+C or Enter to stop visualization...")
    event.set()
    thread.join()
    input("Stopped")

# if __name__ == "__main__":
#     thread.start()
#     sound_thread.start()
#     input("Press CTRL+C or Enter to stop visualization...")
#     event.set()
#     sound_event.set()
#     thread.join()
#     sound_thread.join()
#     input("Stopped")