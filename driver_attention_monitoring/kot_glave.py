import cv2 as cv
import numpy as np
import time

leva_oka = [(0,0)]
desna_oka = [(0,0)]
model2 = "IMPORTAJ MODEL"


color_map = {
    "odprte" : (0, 255, 0), #zelena
    "zaprte" : (0, 0, 255) #rdeca
}

def getContours(img, imgContour):
    global leva_oka, desna_oka
    contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    for cnt in contours:
        area = cv.contourArea(cnt)

        if area > 30:
            peri = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 0.02*peri, True)
            #if len(approx) > 4:
            k=cv.isContourConvex(approx)
            if k:
                cv.drawContours(imgContour, cnt, -1, (0,255,0), 2)

                M = cv.moments(cnt)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                if cX < 20:
                    leva_oka = [(cX, cY)]
                    
                if cX > 20:
                    desna_oka = [(cX, cY)]

def video():
    #input = "../output.mp4"
    input = 1
    capture = cv.VideoCapture(input)

    input_size = (224,224)
    fps = 30
    frame2 = 0
    while True:
        t_start = time.perf_counter()
        ret, frame = capture.read()

        
        results = model2(frame)

        # RESULT pa Frame Funkcija
        boxes = results.xyxy[0].numpy()[:, :4]
        if len(boxes) > 0:

            areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            max_area_index = np.argmax(areas)

            najvecji = results.xyxy[0][max_area_index]
            x1, y1, x2, y2 = map(int,najvecji[:4])
            label = results.names[int(najvecji[-1])]
            #conf = float(najvecji[-2])

            color_box = color_map[label]

            #print(results)
            cv.rectangle(frame, (x1, y1), (x2, y2), color_box, 1)
            cv.putText(frame, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            #cv.putText(frame, str(conf), (x1 - 30, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            #RETURN label
            roi_top_left = (x1, y1)
            roi_bottom_right = (x2, y2)
            roi = frame[roi_top_left[1]:roi_bottom_right[1], roi_top_left[0]:roi_bottom_right[0]]
            
            imgBlur = cv.GaussianBlur(roi, (7, 7), 1)
            imgGray = cv.cvtColor(imgBlur, cv.COLOR_BGR2GRAY)


            imgCanny = cv.Canny(imgGray, 50, 50) # threshold1, threshold2

            #kernal = np.ones((3, 3))
            #imgDil = cv.dilate(imgCanny, kernal, iterations=1)

            getContours(imgCanny, roi)

            delta_x = leva_oka[0][0] - desna_oka[0][0]
            delta_y = leva_oka[0][1] - desna_oka[0][1]
            epsilon = 1e-7 
            angle = np.arctan(delta_y / (delta_x + epsilon))
            angle = (angle * 180) / np.pi
            #print(angle)

            cv.putText(frame, str(angle.round(3)) + "%", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            

            cv.waitKey(1)


        cv.imshow("Video", frame)


        if cv.waitKey(5) & 0xFF == ord('q'):
            break

    capture.release()
    cv.destroyAllWindows()