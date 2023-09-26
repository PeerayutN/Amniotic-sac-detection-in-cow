from ultralytics import YOLO
import cv2
import math
import numpy as np
from time import time

# Load video
cap = cv2.VideoCapture("dataset/calving (1).mp4")  # For video

# Load Model
model = YOLO("best_n.pt")
classNames = ["Amniotic-sac"]

# Detecting objects and showing informations on the screen 
while True:
    success, img = cap.read()
    if success == True:
        start_time = time()
        img_GRAY = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        cv2.imwrite("frame.png",img_GRAY)

        # Predict on image 
        results = model(source="frame.png", conf=0.5)
        for r in results:
            boxes = r.boxes
            for box in boxes:
            # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            # Confidence
                conf = math.ceil((box.conf[0] * 100))
            # Class Name
                cls = int(box.cls[0])
                cv2.putText(img,f'Sac : {conf}%', (x1, y1-10),cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,255,255), 2)

            # FPS calculate
            end_time = time()
            fps = round(1/np.round(end_time - start_time, 2),2)
            cv2.putText(img, f'FPS: {(fps)}', (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    img = cv2.resize(img, (960, 540)) # Resize video show  
    cv2.imshow("CalfuL", img)
    key = cv2.waitKey(1) # Press Esc button to stop processing
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

