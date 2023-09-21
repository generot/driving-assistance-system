#!/usr/bin/python

import cv2
import numpy as np
import math

from detect import classify_car_rear

FPS = 60

n_disp = 64
block_sz = 7
sigma = 1.5
lmb = 8000

#OOI - Object Of Interest (tuple: (width-px, height-px))
def calibrate_camera(frame, OOI, real_measurements, dist_from_cam):
    dims = frame.shape

    cx = dims[1] // 2
    cy = dims[0] // 2

    fx = math.floor(dist_from_cam * OOI[0] / real_measurements[0])
    fy = math.floor(dist_from_cam * OOI[1] / real_measurements[1])

    #Camera intrinsic matrix
    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ])

def create_stereo_matcher():
    left = cv2.StereoBM_create(numDisparities=n_disp, blockSize=block_sz)
    right = cv2.ximgproc.createRightMatcher(left)

    wls = cv2.ximgproc.createDisparityWLSFilter(left)
    wls.setLambda(lmb)
    wls.setSigmaColor(sigma)

    return (left, right, wls)

def compute_disparity(frame1, frame2):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    left, right, wls = create_stereo_matcher()

    left_disp = left.compute(gray1, gray2)
    right_disp = right.compute(gray2, gray1)

    disparity = wls.filter(left_disp, gray1, disparity_map_right=right_disp)
    normalized = cv2.normalize(disparity, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return (disparity, normalized)

def stereo_image(path1, path2):
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)

    #Real measurements in [cm]
    intr_mat = calibrate_camera(img1, (100, 100), (155, 155), 600)

    #Imaginary setup
    fx = intr_mat[0,0]
    d = 50 #cm

    cropped, results = classify_car_rear(img1, (100, 300), (350, 600))

    #disparity, normalized = compute_disparity(img1, img2)

    for rect in results:
        x, y, w, h = rect
        detected = cropped[y : y + h, x : x + w]

        match = cv2.matchTemplate(img2, detected, cv2.TM_CCOEFF)
        min_val, max_val, min_lc, max_lc = cv2.minMaxLoc(match)

        upper_left = max_lc

        disparity = abs(upper_left[0] - (x + 350))
        depth = (fx * d / disparity) / 100 #m

        cv2.rectangle(cropped, (x, y), (x + w, y + h), (0, 255, 0))

        cv2.putText(img1, f"Distance: {depth:.3} m", (x + 350, y + 100 - 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    
    #cv2.imshow("Disparity", normalized)
    cv2.imshow("Left Original", img1)
    cv2.waitKey(0)


def stereo_video(path1, path2):
    cam1 = cv2.VideoCapture(path1)
    cam2 = cv2.VideoCapture(path2)

    while cam1.isOpened() and cam2.isOpened():
        retcode, frame1 = cam1.read()
        retcode, frame2 = cam2.read()

        if cv2.waitKey(1000 // FPS) == ord('e'):
            break

        frame1 = cv2.resize(frame1, (640, 480))
        frame2 = cv2.resize(frame2, (640, 480))
        #frame = cv2.hconcat((frame1, frame2))

        disparity, normalized = compute_disparity(frame1, frame2)
    
        cv2.imshow("Disparity", normalized)

        cv2.imshow("Cam 1", frame1)
        cv2.imshow("Cam 2", frame2)

        #cv2.imshow("Stereo", frame)

    cam1.release()
    cam2.release()

    cv2.destroyAllWindows()

def stereo_video_test(joined_frame):
    dims = joined_frame.shape

    left_frame  = joined_frame[0 : dims[0], 0  : dims[1] // 2]
    right_frame = joined_frame[0 : dims[0], dims[1] // 2 : dims[1]]

    disparity, normalized = compute_disparity(left_frame, right_frame)

    cv2.imshow("Disparity", normalized)
    #cv2.imshow("Left frame", left_frame)
    #cv2.imshow("Right frame", right_frame)

def read_video(path, callback):
    FPS = 20
    vid = cv2.VideoCapture(path)

    while vid.isOpened():
        retcode, frame = vid.read()

        if cv2.waitKey(1000 // FPS) == ord('e'):
            break

        callback(frame)

def main():
    #stereo_image("../samples/stereo/ambush_5_left.jpg","../samples/stereo/ambush_5_right.jpg")
    #stereo_video("../samples/stereo/test_left.mp4", "../samples/stereo/test_right.mp4")

    #Video from: 
    #https://github.com/introlab/rtabmap/wiki/Stereo-mapping#process-a-side-by-side-stereo-video-with-calibration-example
    read_video("../samples/stereo/conjoined_stereo.avi", stereo_video_test)

    #stereo_image("../samples/stereo/left_view_car.bmp","../samples/stereo/right_view_car.bmp")

    return 0

if __name__ == "__main__":
    main()
