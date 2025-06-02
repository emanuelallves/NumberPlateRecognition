from ultralytics import YOLO
import cv2 as cv
from sort.sort import *
import util
import numpy as np

mot_tracker = Sort()

results = {}

coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('runs/detect/train6/weights/best.pt')

cap = cv.VideoCapture('plate_test.mp4')

vehicles = [2, 3, 5, 7]

ret = True
frame_nmr = -1

while ret:
     frame_nmr += 1
    
     ret, frame = cap.read()

     if ret:
          results[frame_nmr] = {}
          detections = coco_model(frame)[0]
          detections_ = []

          for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection

            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

          track_ids = mot_tracker.update(np.asarray(detections_))

          license_plates = license_plate_detector(frame)[0]

          for license_plate in license_plates.boxes.data.tolist():
               x1, y1, x2, y2, score, class_id = license_plate

               if score >= 0.4:
                    xcar1, ycar1, xcar2, ycar2, car_id = util.get_car(license_plate, track_ids)

                    if car_id != -1:
                         license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

                         license_plate_crop_gray = cv.cvtColor(license_plate_crop, cv.COLOR_BGR2GRAY)

                         _, license_plate_crop_thresh = cv.threshold(license_plate_crop_gray, 64, 255, cv.THRESH_BINARY_INV)

                         license_plate_text, license_plate_text_score = util.read_license_plate(license_plate_crop_thresh)

                         if license_plate_text is not None:
                              results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                            'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                           'text': license_plate_text,
                                                                           'bbox_score': score,
                                                                           'text_score': license_plate_text_score}}

util.write_csv(results, 'test.csv')