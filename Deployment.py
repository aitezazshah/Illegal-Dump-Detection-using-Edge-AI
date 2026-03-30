{\rtf1\ansi\ansicpg1252\cocoartf2868
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 from ultralytics import YOLO\
from picamera2 import Picamera2\
import cv2\
\
# Load models\
garbage_model = YOLO("/home/huzaifa/Downloads/best.onnx")\
person_model = YOLO("yolov8n.pt")\
\
# Start camera\
picam2 = Picamera2()\
picam2.configure(picam2.create_preview_configuration(\
    main=\{"size": (320, 320)\}\
))\
picam2.start()\
\
frame_count = 0\
g_results = None\
p_results = None\
\
while True:\
    frame = picam2.capture_array()\
    frame_count += 1\
\
    # Resize (important for speed)\
    frame = cv2.resize(frame, (256, 256))\
\
    # Garbage detection every 2 frames\
    if frame_count % 2 == 0:\
        g_results = garbage_model(frame, imgsz=256)\
\
    # Person detection every 5 frames\
    if frame_count % 5 == 0:\
        p_results = person_model(frame, imgsz=256)\
\
    img = frame.copy()\
\
    garbage_detected = False\
    person_detected = False\
\
    # Draw garbage detections\
    if g_results is not None:\
        for box in g_results[0].boxes:\
            garbage_detected = True\
            x1, y1, x2, y2 = map(int, box.xyxy[0])\
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)\
            cv2.putText(img, "Garbage", (x1, y1 - 10),\
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)\
\
    # Draw person detections\
    if p_results is not None:\
        for box in p_results[0].boxes:\
            cls = int(box.cls[0])\
            if p_results[0].names[cls] == "person":\
                person_detected = True\
                x1, y1, x2, y2 = map(int, box.xyxy[0])\
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)\
                cv2.putText(img, "Person", (x1, y1 - 10),\
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)\
\
    # Illegal dumping logic\
    if garbage_detected and person_detected:\
        cv2.putText(img, "Illegal Dumping!",\
                    (20, 40),\
                    cv2.FONT_HERSHEY_SIMPLEX,\
                    0.8,\
                    (0, 0, 255),\
                    2)\
\
    # Show live feed\
    cv2.imshow("Live Detection", img)\
\
    # Exit on ESC\
    if cv2.waitKey(1) == 27:\
        break\
\
cv2.destroyAllWindows()}