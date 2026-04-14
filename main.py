import cv2
import torch
import easyocr
import datetime
import os
import numpy as np
from ultralytics import YOLO

MODEL_PATH = 'runs/detect/train/weights/best.pt' 
VIDEO_SOURCE = 'test_videos/video-5-cut.mp4' 
LOG_FILE = "traffic_logs.csv"

ZONE_TOP = 300
ZONE_BOTTOM = 950

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        f.write("Timestamp,Plate,Direction\n")

print("::::STARTING MOODELS:::::")
model = YOLO(MODEL_PATH)
reader = easyocr.Reader(['en'], gpu=True)
logged_ids = set()

def preprocess_plate(img):
    if img is None or img.size == 0: return None
    img = cv2.resize(img, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(gray, -1, kernel)

def clean_plate_text(text):
    clean = "".join(e for e in text if e.isalnum()).upper()
    
    if len(clean) >= 4:
        if clean.startswith(('6', 'C', 'G', '0')):
            return "GJ" + clean[2:]
        return clean
    return None

cap = cv2.VideoCapture(VIDEO_SOURCE)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    results = model.track(frame, persist=True, conf=0.3, verbose=False)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)
            
            if ZONE_TOP < y2 < ZONE_BOTTOM:
                if track_id not in logged_ids:
                    
                    h = y2 - y1
                    crop_y_start = y1 + int(h * 0.50) 
                    plate_img = frame[crop_y_start:y2, x1:x2]
                    
                    processed = preprocess_plate(plate_img)
                    if processed is not None:
                        
                        cv2.imshow("AI_SEEING_PLATE", processed)
                        
                        ocr_results = reader.readtext(processed, detail=0)
                        
                        if ocr_results:
                            print(f":::::VEHICLE {track_id} OCR Raw: {ocr_results}")
                            
                            for text in ocr_results:
                                plate_no = clean_plate_text(text)
                                
                                if plate_no:
                                    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    print(f":::::WRITING TO CSV: {plate_no}")
                                    
                                    with open(LOG_FILE, "a") as f:
                                        f.write(f"{now},{plate_no},IN\n")
                                    
                                    logged_ids.add(track_id)
                                    break 

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{track_id}", (x1, y1-10), 0, 0.5, (0, 255, 0), 2)

    cv2.line(frame, (0, ZONE_TOP), (frame.shape[1], ZONE_TOP), (255, 255, 255), 1)
    cv2.line(frame, (0, ZONE_BOTTOM), (frame.shape[1], ZONE_BOTTOM), (255, 255, 255), 1)
    
    cv2.imshow(":::::GUJARAT VEHICLE MONITORING:::::", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()