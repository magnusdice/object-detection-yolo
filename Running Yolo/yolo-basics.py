# TEST IF YOLO WORKS
from ultralytics import YOLO
import cv2

model = YOLO('../yolo-weights/yolov8n.pt')
results = model('Running Yolo/images/1.jpg',show = True)
cv2.waitKey(0)



#https://www.youtube.com/watch?v=WgPbbWmnXJ8&t=156s