#!../venv/bin/python

import cv2
import numpy as np
import re

from glob import glob
#from imutils import contours

#RED (lower boundary)
lower1 = np.array([0, 100, 20])
upper1 = np.array([10, 255, 255])
 
#RED (upper boundary)
lower2 = np.array([160,100,20])
upper2 = np.array([179,255,255])

def recognize_sl_sign(frame, crop_y):
    augmented = frame[crop_y[0] : crop_y[1]]

    hsv_img = cv2.cvtColor(augmented, cv2.COLOR_BGR2HSV)

    #gray = cv2.cvtColor(augmented, cv2.COLOR_BGR2GRAY)
    #ret, threshold = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    lower = cv2.inRange(hsv_img, lower1, upper1)
    upper = cv2.inRange(hsv_img, lower2, upper2)

    filtered = lower + upper

    blurred = cv2.GaussianBlur(filtered, (7, 7), 5)
    dilated = cv2.dilate(blurred, (11, 11))
    circles = cv2.HoughCircles(dilated, cv2.HOUGH_GRADIENT, 2, len(augmented) // 4, param1=220, param2=60, maxRadius=50, minRadius=10)

    return augmented, dilated, circles

def recognize_sign_digits(knn, gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    threshold = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 4)

    cnts, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    filtered_contours = [cv2.boundingRect(x) for x in cnts if cv2.contourArea(x) > 100.0]
    sorted_contours = sorted(filtered_contours, key=lambda i: i[0])

    digits = []

    for cnt in sorted_contours:
        x, y, w, h = cnt

        digit_frame = threshold[y : y + h, x : x + w]

        resized = cv2.resize(digit_frame, (20, 20))
        reshaped = resized.reshape(-1, 400).astype(np.float32)
    
        _, result, _, _ = knn.findNearest(reshaped, 7)
        int_result = np.unique(result).astype(np.int32)

        digits.append(str(int_result[0]))
        print(int_result)

    return "".join(digits)

def read_train_data(dataset_path, bounds):
    def read_and_resize(path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return cv2.resize(img, (20, 20))

    paths = glob(dataset_path)

    image_paths = { path: glob(f"{path}/*")[bounds[0] : bounds[1]] for path in paths }
    images = { re.findall(r".*?(\d+)", path)[0] : [read_and_resize(name) for name in image_paths[path]] for path in paths }

    return images

def train_knn():
    #bounds = (30, 200)
    bounds = (0, 3)
    #images = read_train_data("../models/signs/gtsrb/*", bounds)
    images = read_train_data("../models/signs/digits/*", bounds)

    train_labels = list(images.keys())

    del train_labels[4]
    del train_labels[8]

    train_data = [images[key] for key in train_labels]

    train = np.array(train_data).reshape(-1, 400).astype(np.float32)
    train_labels = np.array(train_labels).repeat(bounds[1] - bounds[0]).astype(np.float32)[:,np.newaxis]

    knn = cv2.ml.KNearest.create()
    knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)

    return knn


def save_for_training(digit_frame):
    some_arr = np.arange(10)

    cv2.imshow("Digit", digit_frame)
    np.random.shuffle(some_arr)

    ident = chr(cv2.waitKey(0))
    cv2.imwrite(f"../models/signs/digits/{ident}/" + "".join(some_arr.astype(str)) + ".png", digit_frame)


def upper_limit_rec():
    img = cv2.imread(cv2.samples.findFile("../samples/sl_sign2.jpg"))
    #img = cv2.imread("../models/signs/gtsrb/120/00008_00001_00027.png")
    #img = cv2.imread("../models/signs/gtsrb/60/00003_00032_00013.png")
    #img = cv2.imread("../models/signs/gtsrb/100/00007_00042_00023.png")
    img = cv2.convertScaleAbs(img, alpha=0.7, beta=30)

    knn = train_knn()
    augmented, dilated, circles = recognize_sl_sign(img, (0, img.shape[0]))

    if np.any(circles) != None:
        for i, circle in enumerate(circles):
            for x, y, r in circle:
                scale = 1.5

                round_x = int(round(x))
                round_y = int(round(y))
                round_r = int(round(r / scale))

                square_frame = img[
                    round_y - round_r : round_y + round_r, 
                    round_x - round_r : round_x + round_r
                ]

                resz = cv2.resize(square_frame, (50, 50))
                gray = cv2.cvtColor(resz, cv2.COLOR_BGR2GRAY)
                
                '''
                resized = cv2.resize(gray, (20, 20))
                reshaped = resized.reshape(-1, 400).astype(np.float32)
    
                _, result, _, _ = knn.findNearest(reshaped, 7)
                '''

                cv2.circle(img, (round_x, round_y), round_r, (0, 255, 0), 2)
                #print(np.unique(result, return_counts=True))
                
                #resized = cv2.resize(gray, (50, 50))
                blur = cv2.GaussianBlur(gray, (5, 5), 1)
                #_, threshold = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY_INV)
                threshold = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 4)
                #edges = cv2.Canny(threshold, 50, 150)

                cnts, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                filtered_contours = [cv2.boundingRect(x) for x in cnts if cv2.contourArea(x) > 100.0]
                sorted_contours = sorted(filtered_contours, key=lambda i: i[0])

                print(sorted_contours)
                
                for cnt in sorted_contours:
                    x, y, w, h = cnt

                    #cv2.rectangle(threshold, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    digit_frame = threshold[y : y + h, x : x + w]

                    resized = cv2.resize(digit_frame, (20, 20))
                    reshaped = resized.reshape(-1, 400).astype(np.float32)
    
                    _, result, _, _ = knn.findNearest(reshaped, 7)
                    print(np.unique(result))

                    #save_for_training(digit_frame)
    
                    cv2.destroyAllWindows()

                #cv2.imshow(f"Circle #{i}", threshold)
                
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    
    cv2.imshow("Traffic Sign", img)
    cv2.imshow("Traffic Sign - Dilated", dilated)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    upper_limit_rec()
