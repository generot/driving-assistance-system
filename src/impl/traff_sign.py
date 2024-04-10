#!../venv/bin/python

import cv2
import numpy as np
import os

from glob import glob

#KNN Training data
KNN_IMAGE_SZ = (20, 20)
KNN_DATASET_PATH = "../models/signs/digits/"

#RED (lower boundary)
lower1 = np.array([0, 100, 20])
upper1 = np.array([10, 255, 255])
 
#RED (upper boundary)
lower2 = np.array([160,100,20])
upper2 = np.array([179,255,255])

sign_gauss_blur_kernel = (7,7)
sign_gauss_blur_delta = 5
sign_dilation_kernel = (11, 11)
sign_hcircles_p1 = 220
sign_hcircles_p2 = 60
sign_hcircles_max_r = 40
sign_hcircles_min_r = 10

def recognize_sl_sign(frame: cv2.Mat, crop_y: tuple[int, int] = None) -> tuple[cv2.Mat, cv2.Mat, cv2.Mat]:
    augmented = frame[crop_y[0] : crop_y[1]] if crop_y != None else frame

    hsv_img = cv2.cvtColor(augmented, cv2.COLOR_BGR2HSV)

    lower = cv2.inRange(hsv_img, lower1, upper1)
    upper = cv2.inRange(hsv_img, lower2, upper2)

    filtered = lower + upper

    blurred = cv2.GaussianBlur(filtered, sign_gauss_blur_kernel, sign_gauss_blur_delta)
    dilated = cv2.dilate(blurred, sign_dilation_kernel)

    circles = cv2.HoughCircles(dilated, 
                               cv2.HOUGH_GRADIENT, 
                               2, 
                               len(augmented) // 4, 
                               param1=sign_hcircles_p1, 
                               param2=sign_hcircles_p2, 
                               maxRadius=sign_hcircles_max_r, 
                               minRadius=sign_hcircles_min_r)

    return augmented, dilated, circles

def recognize_sign_digits(knn: cv2.ml.KNearest, gray: cv2.Mat) -> str:
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    threshold = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 4)

    cnts, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    filtered_contours = [cv2.boundingRect(x) for x in cnts if cv2.contourArea(x) > 90.0]
    sorted_contours = sorted(filtered_contours, key=lambda i: i[0])

    digits = []

    for cnt in sorted_contours:
        x, y, w, h = cnt

        digit_frame = threshold[y : y + h, x : x + w]

        resized = cv2.resize(digit_frame, KNN_IMAGE_SZ)
        reshaped = resized.reshape(-1, KNN_IMAGE_SZ[0] * KNN_IMAGE_SZ[1]).astype(np.float32)
    
        _, result, _, _ = knn.findNearest(reshaped, 12)
        int_result = np.unique(result).astype(np.int32)

        digits.append(str(int_result[0]))
        #print(int_result)

    return "".join(digits)

def read_train_data(dataset_path: str) -> tuple[list[int], list[cv2.Mat]]:
    def read_and_resize(path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return cv2.resize(img, KNN_IMAGE_SZ)

    paths = glob(dataset_path)
    
    labels = []
    images = []

    for i, path in enumerate(paths):
        globbed_path = os.path.join(path, "*")
        image_paths = glob(globbed_path)

        for img_path in image_paths:
            labels.append(i)
            images.append(read_and_resize(img_path))

    return labels, images

def train_knn(dataset_path: str) -> cv2.ml.KNearest:
    globbed_path = os.path.join(dataset_path, "*")
    train_labels, train_data = read_train_data(globbed_path)

    train_data_flattened = np.array(train_data).reshape(-1, KNN_IMAGE_SZ[0] * KNN_IMAGE_SZ[1]).astype(np.float32)
    train_labels = np.array(train_labels).astype(np.float32)[:,np.newaxis]

    knn = cv2.ml.KNearest.create()
    knn.train(train_data_flattened, cv2.ml.ROW_SAMPLE, train_labels)

    return knn


def test_video():
    video = cv2.VideoCapture("../samples/private/sample_carigradsko_1.mp4")
    knn = train_knn(KNN_DATASET_PATH)

    while video.isOpened():
        ret, frame = video.read()

        if ret == False:
            print("Could not read frame from video.")
            break

        frame = cv2.resize(frame, (1280, 720))
        aug, dilated, circles = recognize_sl_sign(frame)

        limit = 0
        prev_limit = 0

        if np.any(circles):
            for circle in circles[0]:
                x, y, r = circle

                x_int = int(x)
                y_int = int(y)
                r_int = int(r)

                sign_frame = frame[y_int - r_int : y_int + r_int, 
                                   x_int - r_int : x_int + r_int]

                cv2.circle(aug, (x_int, y_int), r_int, (0, 255, 0), 1)

                if np.any(sign_frame):
                    prev_limit = limit
                    limit = recognize_sign_digits(knn, cv2.cvtColor(sign_frame, cv2.COLOR_BGR2GRAY))    

        if limit != prev_limit:
            print(f"Detected Limit: {limit}")

        cv2.imshow("Video", aug)
        cv2.imshow("Dilated", dilated)
    
        if cv2.waitKey(1000 // 60) == ord("e"):
            cv2.destroyAllWindows()
            break

    video.release()

def test_main():
    img = cv2.imread("../samples/sl_sign2.jpg")
    frame, dilated, circles = recognize_sl_sign(img)

    for circle in circles[0]:
        x, y, r = circle
        x_int = int(x)
        y_int = int(y)
        r_int = int(r)

        cv2.circle(frame, (x_int, y_int), r_int, (0, 255, 0), 2)

    cv2.imshow("Image", frame)
    cv2.imshow("Dilated", dilated)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    #test_main()
    test_video()