#!../venv/bin/python

import cv2
import numpy as np

from triangulation import linear_LS_triangulation

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
