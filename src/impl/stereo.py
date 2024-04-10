#!../venv/bin/python

import cv2
import numpy as np
import math

from triangulation import linear_LS_triangulation

FPS = 20

n_disp = 160
block_sz = 21
sigma = 1.5
lmb = 8000

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


def match_roi(img_to_match, roi_frame, detection_box):
    x, y, w, h = detection_box

    if w == 0 or h == 0:
        return None

    detection = roi_frame[y : y + h, x : x + w]

    match = cv2.matchTemplate(img_to_match, detection, cv2.TM_CCOEFF)

    cv2.normalize(match, match, 0, 1, cv2.NORM_MINMAX, -1)
    min_val, max_val, min_lc, max_lc = cv2.minMaxLoc(match)

    upper_left = (max_lc[0], max_lc[1])

    return upper_left, w, h

def project_to_3d(camera_mats, detection_left_pt, detection_right_pt):
    camera_mat_1 = camera_mats["p1"]
    camera_mat_2 = camera_mats["p2"]

    proj_points_left = np.array([ [*detection_left_pt] ])
    proj_points_right = np.array([ [*detection_right_pt] ])

    #points_4d = cv2.triangulatePoints(camera_mat_1, camera_mat_2, proj_points_left, proj_points_right)
    points_3d, _ = linear_LS_triangulation(proj_points_left, camera_mat_1, proj_points_right, camera_mat_2)

    return points_3d

def get_world_dist(camera_mats, point_cm_3d):
    T = camera_mats["tvecs"]
    baseline_world = 20 #cm

    correction = 2 #compensating for calibration errors
    error = 20 #After testing, it was determined that the calculated distance is off by about 20 cm

    baseline_cmspace = abs(T[0,0]) + correction
    point_z_cmspace = point_cm_3d[0,2]

    point_z_world = baseline_world * point_z_cmspace // baseline_cmspace - error

    return point_z_world
