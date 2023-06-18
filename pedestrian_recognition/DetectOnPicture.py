import getopt
import math
import sys
import os

import cv2
from ultralytics import YOLO


def DetectPicture(input_photo_path, output_photo_path):
    # Load YOLOv8 model
    model = YOLO("yolov8n.pt")

    frame = cv2.imread(input_photo_path)

    results = model.predict(frame, conf=0.59, classes=[0], verbose=False)[0].numpy()

    # Draw each box from results directly
    for param in results:
        for box in param.boxes.data:
            box_size = math.dist((box[0], box[1]), (box[2], box[3]))
            cv2.putText(frame, "Person " + str(round(box[4], 3)),
                        (int(box[0]), int(box[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            if box_size > 180:
                cv2.rectangle(frame, (int(box[0]), int(box[1])),
                              (int(box[2]), int(box[3])), (0, 0, 255), 2)
            elif box_size > 100:
                cv2.rectangle(frame, (int(box[0]), int(box[1])),
                              (int(box[2]), int(box[3])), (0, 165, 255), 2)
            else:
                cv2.rectangle(frame, (int(box[0]), int(box[1])),
                              (int(box[2]), int(box[3])), (0, 255, 0), 2)

    # Write frame to output
    cv2.imwrite(output_photo_path, frame)

argv = sys.argv[1:]

# pass argument to the program if video input or camera input
opts, args = getopt.getopt(argv, "hi:", ["input="])
path = None

for opt, arg in opts:
    if opt == '-h':
        print('DetectOnPicture.py -i <picture_path>')
        sys.exit()
    elif opt == '-i':
        path = arg

if path is None:
    print('Pot ni podana!')
    sys.exit()

input_video_path = os.path.join("Data", path)
output_video_path = os.path.join("Output", path)

print(f"Working on {path}..")

DetectPicture(input_video_path, output_video_path)

print("saved to " + output_video_path)
print(f"Done with {path}..\n")