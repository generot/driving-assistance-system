#!../venv/bin/python

import cv2
import numpy as np
import os
import math

from glob import glob

NO_DETECT = -1

#KNN Training data
KNN_IMAGE_SZ = (20, 20)
KNN_DATASET_PATH = "../models/signs/digits/"

sign_gauss_blur_kernel = (7,7)
#sign_gauss_blur_delta = 5
sign_gauss_blur_delta = 1.5
sign_dilation_kernel = (11, 11)
sign_hcircles_p1 = 220
sign_hcircles_p2 = 60
sign_hcircles_max_r = 80
sign_hcircles_min_r = 30

binary_thresh = 160
area_thresh = 3000

def check_circularity(cnt):
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    
    if perimeter == 0:
        return False, area

    circularity = 4*math.pi*(area/(perimeter*perimeter))

    if 0.7 < circularity < 1:
        return True, area, cnt
    
    return False, area

def recognize_sl_sign(frame: cv2.Mat, crop_y: tuple[int, int] = None):
    augmented = frame[crop_y[0] : crop_y[1]] if crop_y != None else frame
    
    blurred = cv2.medianBlur(augmented, 3)

    yuv = cv2.cvtColor(blurred, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(yuv)
    
    v1 = cv2.normalize(v, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    _, threshold = cv2.threshold(v1, binary_thresh, 255, cv2.THRESH_BINARY)

    threshold = cv2.dilate(threshold, sign_dilation_kernel, iterations=5)

    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = []

    for con in contours:
        res = check_circularity(con)

        if res[0] == True:
            areas.append(res)

    if areas == []:
        return augmented, threshold, None

    _, _, biggest_cnt = max(areas, key=lambda val: val[1])
    
    return augmented, threshold, biggest_cnt

def recognize_sign_digits(knn: cv2.ml.KNearest, gray: cv2.Mat, frame) -> str:
    imgFloat = frame.astype(np.float32) / 255.
    kChannel = 1 - np.max(imgFloat, axis=2)
    kChannel = (255 * kChannel).astype(np.uint8)

    #threshold = cv2.adaptiveThreshold(kChannel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 8)
    _, threshold = cv2.threshold(kChannel, 150, 255, cv2.THRESH_BINARY)
    dilation = cv2.dilate(threshold, (7, 7))

    h, w = dilation.shape
    h5 = h // 6
    w5 = w // 6

    threshold = threshold[h5 : h - h5, w5 : w - w5]

    cnts, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    filtered_contours = [cv2.boundingRect(x) for x in cnts if cv2.contourArea(x) > 200]
    sorted_contours = sorted(filtered_contours, key=lambda i: i[0])

    cv2.imshow("Contours", threshold)

    digits = []

    for cnt in sorted_contours:
        x, y, w, h = cnt

        digit_frame = threshold[y : y + h, x : x + w]

        x += w5
        y += h5

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)

        resized = cv2.resize(digit_frame, KNN_IMAGE_SZ)
        reshaped = resized.reshape(-1, KNN_IMAGE_SZ[0] * KNN_IMAGE_SZ[1]).astype(np.float32)
    
        _, result, _, _ = knn.findNearest(reshaped, k=12)
        print(result)
        int_result = np.unique(result).astype(np.int32)

        digits.append(str(int_result[0]))

    return "".join(digits)

def deduce_value(recognized_value: str) -> str:
    if not len(recognized_value):
        return NO_DETECT

    if recognized_value[0] == "0" or (val := int(recognized_value)) > 200:
        return NO_DETECT
    
    if val % 5 != 0:
        return NO_DETECT

    return val
            

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

class TSR:
    def __init__(self, knn: cv2.ml.KNearest):
        self.knn = knn
        self.limit = 0
        self.prev_limit = 0
    
    def detect_sl_on_frame(self, frame):
        aug, threshold, circle = recognize_sl_sign(frame)
                
        if np.any(circle) and cv2.contourArea(circle) > area_thresh:
            x, y, w, h = cv2.boundingRect(circle)

            tl = (x, y)
            br = (x + w, y + h)

            sign_frame = frame[y : y + h, x : x + w]
            
            #cv2.rectangle(aug, tl, br, (0, 255, 0), 1)

            self.prev_limit = self.limit
            self.limit = recognize_sign_digits(self.knn, cv2.cvtColor(sign_frame, cv2.COLOR_BGR2GRAY), sign_frame)

            return tl, br
        
        return None, None
        
    def has_sl_changed(self):
        return self.limit != self.prev_limit
    
    def get_limit(self):
        return deduce_value(self.limit)

    

def test_video_2():
    #video = cv2.VideoCapture("../samples/private/sample_carigradsko_1.mp4")
    video = cv2.VideoCapture(0)
    knn = train_knn(KNN_DATASET_PATH)

    while video.isOpened():
        ret, frame = video.read()

        if ret == False:
            print("Could not read frame from video.")
            break

        frame = cv2.resize(frame, (1280, 720))

        aug, threshold, circle = recognize_sl_sign(frame)

        limit = 0
        prev_limit = 0
                
        if np.any(circle) and cv2.contourArea(circle) > area_thresh:
            x, y, w, h = cv2.boundingRect(circle)

            tl = (x, y)
            br = (x + w, y + h)

            sign_frame = frame[y : y + h, x : x + w]
            
            cv2.rectangle(aug, tl, br, (0, 255, 0), 1)

            prev_limit = limit
            limit = recognize_sign_digits(knn, cv2.cvtColor(sign_frame, cv2.COLOR_BGR2GRAY), sign_frame)

            cv2.putText(aug, f"Max limit: {limit}", (tl[0], tl[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 1)

            print(f"Detection: {limit}")

        if limit != prev_limit:
            limit_value = deduce_value(limit)

            if limit_value != NO_DETECT:
                print(f"Detected Limit: {limit_value}")

        cv2.imshow("Video", aug)
        cv2.imshow("Threshold", threshold)
    
        if cv2.waitKey(1000 // 60) == ord("e"):
            cv2.destroyAllWindows()
            break

    video.release()

if __name__ == "__main__":
    #test_main()
    #test_video_2()
    pass