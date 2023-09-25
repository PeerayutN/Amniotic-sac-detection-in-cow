from ultralytics import YOLO
import cv2

model = YOLO('weights/best_n.pt')

result = model(source= "dataset/calving (1).mp4", show=True, conf=0.5)

print(result)

