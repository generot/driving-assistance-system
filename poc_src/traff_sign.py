#!/usr/bin/python

import cv2
import numpy as np

lower_bound = np.array([0, 100, 110])
upper_bound = np.array([19, 255, 255])

#RED (lower boundary)
lower1 = np.array([0, 100, 20])
upper1 = np.array([10, 255, 255])
 
#RED (upper boundary)
lower2 = np.array([160,100,20])
upper2 = np.array([179,255,255])

def recognize_sl_sign(frame, crop_y):
    augmented = frame[crop_y[0]:crop_y[1]]

    hsv_img = cv2.cvtColor(augmented, cv2.COLOR_BGR2HSV)

    #filtered = cv2.inRange(hsv_img, lower_bound, upper_bound)

    lower = cv2.inRange(hsv_img, lower1, upper1)
    upper = cv2.inRange(hsv_img, lower2, upper2)

    full_mask = lower + upper

    filtered = full_mask

    blurred = cv2.GaussianBlur(filtered, (7, 7), 5)
    dilated = cv2.dilate(blurred, (11, 11))
    circles = cv2.HoughCircles(dilated, cv2.HOUGH_GRADIENT, 2, len(augmented) // 4, param1=220, param2=60, maxRadius=40)

    return augmented, dilated, circles

def upper_limit_rec():
    img = cv2.imread(cv2.samples.findFile("../samples/sl_sign2.jpg"))
    img = cv2.convertScaleAbs(img, alpha=0.7, beta=30)

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    filtered = cv2.inRange(hsv_img, lower_bound, upper_bound)
    blurred = cv2.GaussianBlur(filtered, (7, 7), 5)

    '''
    blur1 = cv2.GaussianBlur(b_ch, (7, 7), 5)
    _, binary = cv2.threshold(blur1, 150, 255, cv2.THRESH_BINARY)
    blur2 = cv2.GaussianBlur(binary, (5, 5), 5)
    dilated = cv2.dilate(blurred, (11, 11), iterations=1)
    circles = cv2.HoughCircles(dilated, cv2.HOUGH_GRADIENT, 2, len(img) // 4, param1=300, param2=60, maxRadius=60)
    '''

    dilated = cv2.dilate(blurred, (11, 11))
    circles = cv2.HoughCircles(dilated, cv2.HOUGH_GRADIENT, 2, len(img) // 4, param1=220, param2=60, maxRadius=40)
 
    if np.any(circles) != None:
        for i, circle in enumerate(circles):
            for x, y, r in circle:

                round_x = int(round(x))
                round_y = int(round(y))
                round_r = int(round(r))

                square_frame = img[round_y - round_r : round_y + round_r, round_x - round_r : round_x + round_r]
    
                cv2.circle(img, (round_x, round_y), round_r, (0, 255, 0), 2)

                #cv2.imshow(f"Circle #{i}", square_frame)
                #cv2.waitKey(0)
    
    cv2.imshow("Traffic Sign", img)
    cv2.imshow("Traffic Sign - Dilated", dilated)
