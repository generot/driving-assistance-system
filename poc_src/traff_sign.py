#!/usr/bin/python

import cv2
import numpy as np
import re

from glob import glob

#RED (lower boundary)
lower1 = np.array([0, 100, 20])
upper1 = np.array([10, 255, 255])
 
#RED (upper boundary)
lower2 = np.array([160,100,20])
upper2 = np.array([179,255,255])

def recognize_sl_sign(frame, crop_y):
    augmented = frame[crop_y[0]:crop_y[1]]

    hsv_img = cv2.cvtColor(augmented, cv2.COLOR_BGR2HSV)

    lower = cv2.inRange(hsv_img, lower1, upper1)
    upper = cv2.inRange(hsv_img, lower2, upper2)

    full_mask = lower + upper

    filtered = full_mask

    blurred = cv2.GaussianBlur(filtered, (7, 7), 5)
    dilated = cv2.dilate(blurred, (11, 11))
    circles = cv2.HoughCircles(dilated, cv2.HOUGH_GRADIENT, 2, len(augmented) // 4, param1=220, param2=60, maxRadius=40)

    return augmented, dilated, circles

def train_knn():
    paths = glob("../models/signs/20x20/*.png")

    train_labels = [re.findall(r"(\w+)-1\.png", path)[0] for path in paths]
    train_data = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in paths]

    train = np.array(train_data)
    train = train.reshape(-1, 400).astype(np.float32)

    train_labels = np.array(train_labels).astype(np.float32)[:,np.newaxis]

    knn = cv2.ml.KNearest.create()
    knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)

    return knn

    #test = np.array(train_data[5]).reshape(-1, 400).astype(np.float32)
    #ret, result, nbs, dist = knn.findNearest(test, 3)

def upper_limit_rec():
    img = cv2.imread(cv2.samples.findFile("../samples/sl_sign2.jpg"))
    img = cv2.convertScaleAbs(img, alpha=0.7, beta=30)

    knn = train_knn()
    augmented, dilated, circles = recognize_sl_sign(img, (0, img.shape[0]))

    if np.any(circles) != None:
        for i, circle in enumerate(circles):
            for x, y, r in circle:

                round_x = int(round(x))
                round_y = int(round(y))
                round_r = int(round(r))

                square_frame = img[round_y - round_r : round_y + round_r, round_x - round_r : round_x + round_r]

                gray = cv2.cvtColor(square_frame, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (20, 20))
                reshaped = resized.reshape(-1, 400).astype(np.float32)
    
                _, result, _, _ = knn.findNearest(reshaped, 3)

                cv2.circle(img, (round_x, round_y), round_r, (0, 255, 0), 2)
                print(np.unique(result))

                #cv2.imshow(f"Circle #{i}", square_frame)
                #cv2.waitKey(0)
    
    cv2.imshow("Traffic Sign", img)
    cv2.imshow("Traffic Sign - Dilated", dilated)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

upper_limit_rec()
