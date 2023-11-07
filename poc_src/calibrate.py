import cv2
import glob
import numpy as np

ASPECT_RATIO = 4624 / 3468
CHECKERBOARD_D = (7, 7)

def calibrate_cam(calib_paths):
    for path in calib_paths:
        img = cv2.imread(path)
        img = cv2.resize(img, (720, int(720 / ASPECT_RATIO)))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_D)
        refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

        cv2.drawChessboardCorners(img, CHECKERBOARD_D, refined, ret)
        print(refined)

        cv2.imshow("Pattern", img)
        cv2.waitKey(0)

def main():
    paths = glob.glob("../samples/calibration/*.jpg")

    print(paths)

    calibrate_cam(paths)

if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()