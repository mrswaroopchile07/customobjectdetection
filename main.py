#swaroop
#rootV1
#library import
import cv2
import numpy as np
import time

#Download Yolo from GitHub using Link -  https://pjreddie.com/darknet/yolo/
#Load Yolo
net = cv2.dnn.readNet("yolov3-tiny.weights","yolov3-tiny.cfg")
classes = []   #define classes as array

# Get/ Read Classes names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0,255, size=(len(classes), 3))

#Load Image or Webcam or Video File
cap = cv2.VideoCapture(0)  #web cam pictures
#cap = cv2.VideoCapture("foundry1.mp4")  #Video File Capture
font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()    # this is for FPS calculations
frame_id = 0                   # this is for FPS calculations

while True:                     #while loop to run web cam continuosly
    _, frame = cap.read()
    frame_id += 1
#img = cv2.imread("office-table-and-chairs.png")
    height, width, channels = frame.shape           # frame shape (320,320,3)

    #Detecting Objects Settings (Fixed)
    blob = cv2.dnn.blobFromImage(frame,0.00392,(320,320),(0,0,0),True,crop=False)

    # for b in blob:
    #     for n, img_blob in enumerate(b):
    #         cv2.imshow(str(n), img_blob)

    net.setInput(blob)
    outs = net.forward(outputlayers)

    #Showing Info On Screen
    class_ids = []
    confidences = []
    boxes = []
    detectText_ = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                #Object Detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                #Rectangle Coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                #cv2.rectangle(img,(x,y),(x + w , y + h), (0,255,0),2)
                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    #print(len(boxes))
    #number_objects_detected = len(boxes)
    indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)
    #print(indexes)

    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame,(x,y),(x + w, y + h),color,2)       #bounding box
            cv2.putText(frame,label,(x,y +30),font,1,color,3)       #bounding box name
            cv2.putText(frame, label + " " + str(round(confidence,2)), (x, y + 30), font, 3, color, 3)
            detectText_ = cv2.putText(frame, label + " " + str(round(confidence,2)), (x, y + 30), font, 3, color, 3)
    print(detectText_)
    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time                       # FPS Formula
    cv2.putText(frame, "FPS: " + str(round(fps,2)), (10,30), font, 4,(0,0,0),3)
    cv2.imshow("Image",frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
cv2.destroyAllWindows()
