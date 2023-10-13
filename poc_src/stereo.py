#!/usr/bin/python

import cv2
import numpy as np
import math

from detect import classify_car_rear, average_box_init

FPS = 20

n_disp = 160
block_sz = 21
sigma = 1.5
lmb = 8000

#OOI - Object Of Interest (tuple: (width-px, height-px))
def calibrate_camera(frame_dim, OOI, real_measurements, dist_from_cam):
    cx = frame_dim[1] // 2
    cy = frame_dim[0] // 2

    fx = math.floor(dist_from_cam * OOI[0] / real_measurements[0])
    fy = math.floor(dist_from_cam * OOI[1] / real_measurements[1])

    #Camera intrinsic matrix
    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ])

def create_stereo_matcher():
    left = cv2.StereoSGBM.create(numDisparities=n_disp, blockSize=block_sz)
    left.setSpeckleRange(50)
    left.setSpeckleWindowSize(15)
    #left = cv2.StereoBM_create(numDisparities=n_disp, blockSize=block_sz)
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
    norm = cv2.normalize(left_disp, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return (disparity, norm)


def stereo_image(path1, path2):
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)

    #Real measurements in [cm]
    intr_mat = calibrate_camera(img1.shape, (200, 200), (150, 150), 800)
    global get_average_box

    get_average_box = average_box_init()

    #Imaginary setup
    fx = intr_mat[0,0]
    d = 60 #cm

    stereo(img1, img2, fx, d)
    cv2.waitKey()

def stereo(img1, img2, fx, d):    
    ROI_y = (400, 700)
    ROI_x = (400, 800)

    cropped, results = classify_car_rear(img1, ROI_y, ROI_x)
    #cropped, results = classify_car_rear(img1, (100, 400), (350, 700))

    #disparity, normalized = compute_disparity(img1, img2)

    x, y, w, h = get_average_box([ np.mean(results, axis=0, dtype=np.int32) ])

    #print(results, rect)

    if w > 0 and h > 0:
        detected = cropped[y : y + h, x : x + w]

        match = cv2.matchTemplate(img2, detected, cv2.TM_CCOEFF)

        cv2.normalize(match, match, 0, 1, cv2.NORM_MINMAX, -1)
        min_val, max_val, min_lc, max_lc = cv2.minMaxLoc(match)

        upper_left = (max_lc[0] + 5, max_lc[1])

        disparity = (x + ROI_x[0]) - upper_left[0]
        print(disparity, upper_left, (x + ROI_x[0], y))
        depth = (fx * d / disparity) / 100 #m

        #print(disparity)

        cv2.rectangle(img1, (x + ROI_x[0], y + ROI_y[0]), (x + ROI_x[0] + w, y + ROI_y[0] + h), (0, 0, 255))

        cv2.putText(img1, f"Distance: {depth:.3} m", (x + ROI_x[0], y + ROI_y[0] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    
    #cv2.imshow("Disparity", normalized)
    cv2.imshow("Left Original", img1)
    cv2.imshow("cropped", cropped)
    #cv2.waitKey(0)


def stereo_video(path1, path2, del1=0, del2=0):
    cam1 = cv2.VideoCapture(path1)
    cam2 = cv2.VideoCapture(path2)

    cam1.set(cv2.CAP_PROP_POS_MSEC, del1)
    cam2.set(cv2.CAP_PROP_POS_MSEC, del2)

    mat = calibrate_camera((720, 1280), (506, 305), (170, 140), 200)

    global get_average_box

    get_average_box = average_box_init()

    fx = mat[0, 0] * 2
    d = 18.5 #cm

    while cam1.isOpened() and cam2.isOpened():
        retcode, frame1 = cam1.read()
        retcode, frame2 = cam2.read()

        if cv2.waitKey(1000 // FPS) == ord('e'):
            break

        frame1 = cv2.resize(frame1, (1280, 720))
        frame2 = cv2.resize(frame2, (1280, 720))
        #frame = cv2.hconcat((frame1, frame2))

        #disparity, normalized = compute_disparity(frame1, frame2)
    
        #cv2.imshow("Disparity", normalized)
        stereo(frame1, frame2, fx, d)

        #cv2.imshow("Cam 1", frame1)
        #cv2.imshow("Cam 2", frame2)
        #cv2.waitKey()

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
    #read_video("../samples/stereo/conjoined_stereo.avi", stereo_video_test)
    
    stereo_video("../samples/classified/stereo/2L.mp4", "../samples/classified/stereo/2R.mp4", del1=(26 * 1000 + 500))
    #stereo_image("../samples/stereo/left_view_car.bmp","../samples/stereo/right_view_car.bmp")

    return 0

if __name__ == "__main__":
    main()
