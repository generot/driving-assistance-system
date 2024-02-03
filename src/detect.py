#!../venv/bin/python

import cv2
import numpy as np

from traff_sign import recognize_sl_sign, recognize_sign_digits, train_knn
from rpicam import camera_init, get_frame

FPS = 90

classifier = cv2.CascadeClassifier("../models/cars.xml")
closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

def classify_car_rear(frame, crop_y = None, crop_x = None):
    augmented = frame[crop_y[0]:crop_y[1], crop_x[0]:crop_x[1]] if crop_y and crop_x else frame

    #augmented2 = cv2.convertScaleAbs(augmented, alpha=1.7, beta=40)

    gray = cv2.cvtColor(augmented, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 5)
    dilated = cv2.dilate(blurred, (5, 5), iterations=3)
    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, closing_kernel)

    #detected = classifier.detectMultiScale(closing, 1.08, 7, 0, minSize=(50, 50))
    detected = classifier.detectMultiScale(closing, 1.0006, 3, 0, (150,150))
   
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

        if no_matches_cnt > no_matches_threshold:
            avg_bbox = np.zeros(4, dtype=np.int32)
            no_matches_cnt = 0

        for rect in results:
            x, y, width, height = rect

            rect_arrlike = np.array([x, y, width, height])
    
            if np.array_equal(avg_bbox, np.zeros(4)):
                avg_bbox = np.array(rect_arrlike)
            else:
                avg_bbox = (avg_bbox + rect_arrlike) // 2

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


def main():
    cam0 = camera_init(0)

    last_detected_sl = 0

    get_average_box = average_box_init()
    #knn = train_knn()

    print(cv2.getBuildInformation())

    while True:
        frame = get_frame(cam0)

        #aspect_ratio = frame.shape[1] / frame.shape[0]
        #res_mult = 720

        frame = cv2.convertScaleAbs(frame, alpha=0.8, beta=10)
        #frame = cv2.resize(frame, (int(aspect_ratio * res_mult), res_mult))

        car_frame, result = classify_car_rear(frame, (0, 480), (100, 380))
        avg = get_average_box(result)

        #last_detected_sl = speed_limit_rec(frame)

        #cv2.putText(frame, f"Speed Limit: {last_detected_sl} km / h", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.rectangle(car_frame, (avg[0], avg[1]), (avg[0] + avg[2], avg[1] + avg[3]), (0, 255, 0))

        cv2.imshow("Sample Video", frame)
        #cv2.imshow("Only Red", car_frame)

        if cv2.waitKey(1000 // FPS) == ord('e'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
