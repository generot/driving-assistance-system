import cv2
import numpy as np

from traff_sign import recognize_sl_sign, recognize_sign_digits, train_knn

classifier = cv2.CascadeClassifier("../models/cars.xml")
closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

def classify_car_rear(frame, crop_y = None, crop_x = None):
    augmented = frame[crop_y[0]:crop_y[1], crop_x[0]:crop_x[1]] if crop_y and crop_x else frame

    #augmented2 = cv2.convertScaleAbs(augmented, alpha=1.7, beta=40)

    gray = cv2.cvtColor(augmented, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 5)
    dilated = cv2.dilate(blurred, (5, 5), iterations=3)
    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, closing_kernel)

    #detected = classifier.detectMultiScale(closing, 1.08, 7, 0, minSize=(50, 50))
    #detected = classifier.detectMultiScale(closing, 1.0006, 7, 0, (150,150))
    detected = classifier.detectMultiScale(closing, 1.08, 3, minSize=(50,50))
   
    return (augmented, detected)

def average_box_init():
    no_matches_threshold = 20
    no_matches_cnt = 0

    avg_bbox = np.zeros(4, dtype=np.int32)

    def get_average_box(results):
        nonlocal no_matches_cnt
        nonlocal avg_bbox

        if len(results) == 0:
            no_matches_cnt += 1
        else:
            avg_bbox = np.mean(results, axis=0, dtype=np.int32)

        if no_matches_cnt > no_matches_threshold:
            avg_bbox = np.zeros(4, dtype=np.int32)
            no_matches_cnt = 0

        return avg_bbox

    return get_average_box

def speed_limit_rec(frame):
    sl_frame, dilated, circles = recognize_sl_sign(frame, (200, 600))

    last_detected_sl = 0

    if np.any(circles) != None:
        for i, circle in enumerate(circles):
            for x, y, r in circle:
                round_x = int(round(x))
                round_y = int(round(y))
                round_r = int(round(r))

                square_frame = sl_frame[round_y - round_r : round_y + round_r, round_x - round_r : round_x + round_r]
    
                if 0 not in np.shape(square_frame):
                    gray = cv2.cvtColor(square_frame, cv2.COLOR_BGR2GRAY)
                    resized = cv2.resize(gray, (50, 50))
                        
                    last_detected_sl = recognize_sign_digits(knn, resized)

                    cv2.circle(sl_frame, (round_x, round_y), round_r, (0, 0, 255), 2)

    return last_detected_sl
