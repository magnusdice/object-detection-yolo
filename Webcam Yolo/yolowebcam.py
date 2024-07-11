from ultralytics import YOLO
import cv2
import cvzone #display detection
import math

## FOR WEBCAM ##
# cap = cv2.VideoCapture(0) #initialize the webcam if u have 1 webcam use 0
# cap.set(3,1280) # width = 3, 1280 is resolution
# cap.set(4,720) # height = 4, 720 is resolution

## FOR VIDEO ##
cap = cv2.VideoCapture("Videos/cars.mp4")


model = YOLO('../yolo-weights/yolov8l.pt')

#it will detect based on the id number if 0 then person if 1 then bicycle
# classNames = ["person","bike"]
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
 

while True:
  success, img = cap.read()
  results = model (img, stream=True) # recommended stream = True
  for r in results:
    # get bolding box of each of the results
    boxes = r.boxes 
    for box in boxes:
      # check the x y of the bounding boxes
      # cv2
      x1,y1,x2,y2 = box.xyxy[0]
      x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2),
      cv2.rectangle(img,(x1,y1),(x2,y2),(255,255,0),3) #location , color of the box, thickness

      # cvzone if u want more fancier
      # w,h = x2 - x1 , y1-y2
      # cvzone.cornerRect(img, (w,h))

      ### CONFIDENCE
      ## find the confidence of the box
      # conf = box.conf[0] # not rounded
      conf = math.ceil((box.conf[0]*100))/100 # rounded the confidence by 2 dp using import math
      # print(conf)

      ## place the classification on the middle of the rectangle
      # the img, the what text u want to put at the middle , placement of the text
      # cvzone.putTextRect(img, f'{conf}' ,(max(0,x1),max(35,y1)))

      # class name
      cls = int(box.cls[0]) # convert to integer because it was a floating value
      cvzone.putTextRect(img, f'{classNames[cls]} {conf}' ,(max(0,x1),max(35,y1)),scale = 1,thickness=1)


  cv2.imshow("Image", img)
  cv2.waitKey(1) # 1 ms delay