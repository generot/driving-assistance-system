import cv2
import glob
import numpy as np
import rpicam

CHECKERBOARD_D = (7, 7)

def make_object_points(cboard):
    object_points = np.zeros((cboard[0] * cboard[1], 3), dtype=np.float32)
    object_points[:,:2] = np.mgrid[0:cboard[1], 0:cboard[0]].T.reshape(-1, 2)

    return object_points

def find_chessboard_corners(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_D)
    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

    return ret, refined

def calibrate_cam(calib_paths):
    object_points = make_object_points(CHECKERBOARD_D)

    img_points_1 = []
    img_points_2 = []
    obj_points = []

    for (path_cam0, path_cam1) in calib_paths:
        img0 = cv2.imread(path_cam0)
        img1 = cv2.imread(path_cam1)

        ret1, corners1 = find_chessboard_corners(img0)
        ret2, corners2 = find_chessboard_corners(img1)

        if ret1 == True and ret2 == True:
            img_points_1.append(corners1)
            img_points_2.append(corners2)
            obj_points.append(object_points)

        cv2.drawChessboardCorners(img0, CHECKERBOARD_D, corners1, ret1)
        cv2.drawChessboardCorners(img1, CHECKERBOARD_D, corners2, ret2)

        cv2.imshow("Pattern", img0)
        cv2.imshow("Pattern2", img1)

        cv2.waitKey(0)

    #ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points_2, (rpicam.CAM_WIDTH, rpicam.CAM_HEIGHT), None, None)
    #print(ret)

    return cv2.stereoCalibrate(obj_points,img_points_1, img_points_2, 
                               None, None, None, None,(rpicam.CAM_WIDTH, rpicam.CAM_HEIGHT), flags=0)

def calibrate_stereo():
    paths_cam0 = sorted(glob.glob("../samples/calibration/cm4/cam0/*.jpg"))
    paths_cam1 = sorted(glob.glob("../samples/calibration/cm4/cam1/*.jpg"))

    path_pairs = list(zip(paths_cam0, paths_cam1))

    print(path_pairs)

    ret, Q1, dC1, Q2, dC2, rvecs, tvecs, e_mat, f_mat = calibrate_cam(path_pairs)
    r1, r2, p1, p2, _, _, _ = cv2.stereoRectify(Q1, dC1, Q2, dC2, (rpicam.CAM_WIDTH, rpicam.CAM_HEIGHT), rvecs, tvecs)

    np.savez("../data/stereo_calib_mats.npz", p1=p1, dc1=dC1, p2=p2, dc2=dC2, r1=r1, r2=r2, e=e_mat, f=f_mat)

    print(ret)

if __name__ == "__main__":
    calibrate_stereo()
    cv2.destroyAllWindows()
