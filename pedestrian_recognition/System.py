import cv2
from kafka import KafkaConsumer, KafkaProducer
import numpy as np
import redis


def send_video(video_path, topic):
    producer = KafkaProducer(bootstrap_servers='localhost:9092')

    video = cv2.VideoCapture(video_path)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame_id in range(frame_count):
        ret, frame = video.read()
        if not ret:
            break

        # Serialize the frame and send it as a Kafka message
        frame_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
        producer.send(topic, value=frame_bytes)

    video.release()
    producer.close()


def receive_video(topic):
    consumer = KafkaConsumer(topic, bootstrap_servers='localhost:9092')

    frames = []
    for message in consumer:
        frame_bytes = np.frombuffer(message.value, dtype=np.uint8)
        frame = cv2.imdecode(frame_bytes, cv2.IMREAD_COLOR)
        frames.append(frame)

    consumer.close()
    return frames


def apply_filter(frame, filter_type):
    if filter_type == 'grayscale':
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Add more filters as needed

    return frame


def display_video(frames):
    for frame in frames:
        filtered_frame = apply_filter(frame, 'grayscale')
        cv2.imshow('Video', filtered_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def main():
    video_path = 'path/to/your/video.mp4'
    kafka_topic = 'video-topic'

    # Sending video frames to Kafka
    send_video(video_path, kafka_topic)

    # Receiving video frames from Kafka
    received_frames = receive_video(kafka_topic)

    # Displaying video frames
    display_video(received_frames)


if __name__ == '__main__':
    main()
