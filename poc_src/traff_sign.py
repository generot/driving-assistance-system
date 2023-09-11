#!/usr/bin/python

import cv2
import numpy as np

def upper_limit_rec():
    img = cv2.imread(cv2.samples.findFile("../samples/sl_sign2.jpg"))

    b_ch, g_ch, r_ch = cv2.split(img)

    blur1 = cv2.GaussianBlur(b_ch, (7, 7), 5)
    _, binary = cv2.threshold(blur1, 150, 255, cv2.THRESH_BINARY)
    blur2 = cv2.GaussianBlur(binary, (5, 5), 5)

    dilated = cv2.dilate(blur2, (11, 11), iterations=1)

    circles = cv2.HoughCircles(dilated, cv2.HOUGH_GRADIENT, 2, len(img) // 4, param1=300, param2=60, maxRadius=60)
 
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

upper_limit_rec()

if cv2.waitKey(0) == ord('e'):
    cv2.destroyAllWindows()
