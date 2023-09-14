#!/usr/bin/python

import cv2
import numpy as np
from matplotlib import pyplot as plt

FPS = 60

n_disp = 0
block_sz = 21
sigma = 1.5
lmb = 8000

def create_stereo_matcher():
    left = cv2.StereoBM_create(numDisparities=n_disp, blockSize=block_sz)
    left.setTextureThreshold(10)

    right = cv2.ximgproc.createRightMatcher(left)

    wls = cv2.ximgproc.createDisparityWLSFilter(left)
    wls.setLambda(lmb)
    wls.setSigmaColor(sigma)

    return (left, right, wls)

def stereo_image(path1, path2):
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    d = 12 #cm

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    print(gray1.shape)

    #stereoBM = cv2.StereoBM.create(numDisparities=16, blockSize=15)
    #stereoBM.setMinDisparity(4)
    #stereoBM.setSpeckleRange(16)
    #stereoBM.setSpeckleWindowSize(45)

    left, right, wls = create_stereo_matcher()

    left_disp = left.compute(gray1, gray2)
    right_disp = right.compute(gray2, gray1)

    disparity = wls.filter(left_disp, gray1, disparity_map_right=right_disp)
    depth = (950 * d) / disparity

    normalized = cv2.normalize(disparity, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm2 = cv2.normalize(left_disp, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    cv2.imshow("Disparity", normalized)
    cv2.imshow("Depth", norm2)
    cv2.imshow("Left Original", img1)
    cv2.waitKey(0)

def stereo_video(path1, path2):
    cam1 = cv2.VideoCapture(path1)
    cam2 = cv2.VideoCapture(path2)

    while cam1.isOpened() and cam2.isOpened():
        retcode, frame1 = cam1.read()
        retcode, frame2 = cam2.read()

        frame1 = cv2.resize(frame1, (640, 480))
        frame2 = cv2.resize(frame2, (640, 480))
        #frame = cv2.hconcat((frame1, frame2))

        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        if cv2.waitKey(1000 // FPS) == ord('e'):
            break

        left, right, wls = create_stereo_matcher()

        left_disp = left.compute(gray1, gray2)
        right_disp = right.compute(gray2, gray1)

        #disparity = wls.filter(left_disp, gray1, disparity_map_right=right_disp)

        #stereoBM = cv2.StereoBM.create(numDisparities=16, blockSize=17)
        #disparity = stereoBM.compute(gray1, gray2)

        #normalized = cv2.normalize(disparity, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        norm = cv2.normalize(right_disp, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        cv2.imshow("Disparity", norm)

        cv2.imshow("Cam 1", gray1)
        cv2.imshow("Cam 2", gray2)

        #cv2.imshow("Stereo", frame)

    cam1.release()
    cam2.release()

    cv2.destroyAllWindows()

stereo_image("../samples/stereo/ambush_5_left.jpg","../samples/stereo/ambush_5_right.jpg")
#stereo_video("../samples/stereo/test_left.mp4", "../samples/stereo/test_right.mp4")
