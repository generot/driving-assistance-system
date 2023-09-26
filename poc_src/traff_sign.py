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

def read_train_data(dataset_path, bounds):
    def read_and_resize(path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return cv2.resize(img, (20, 20))

    paths = glob(dataset_path)

    image_paths = { path: glob(f"{path}/*")[bounds[0] : bounds[1]] for path in paths }
    images = { re.findall(r".*?(\d+)", path)[0] : [read_and_resize(name) for name in image_paths[path]] for path in paths }

    return images

def train_knn():
    bounds = (30, 180)
    images = read_train_data("../models/signs/gtsrb/*", bounds)

    train_labels = list(images.keys())
    train_data = [images[key] for key in train_labels]

    train = np.array(train_data).reshape(-1, 400).astype(np.float32)
    train_labels = np.array(train_labels).repeat(bounds[1] - bounds[0]).astype(np.float32)[:,np.newaxis]

    knn = cv2.ml.KNearest.create()
    knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)

    return knn

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
    
                _, result, _, _ = knn.findNearest(reshaped, 7)

                cv2.circle(img, (round_x, round_y), round_r, (0, 255, 0), 2)
                print(np.unique(result, return_counts=True))

                #cv2.imshow(f"Circle #{i}", square_frame)
                #cv2.waitKey(0)
    
    cv2.imshow("Traffic Sign", img)
    cv2.imshow("Traffic Sign - Dilated", dilated)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

#upper_limit_rec()
