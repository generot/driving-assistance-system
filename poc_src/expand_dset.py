import json
import cv2
import os
import random
import supervisely as sly
from matplotlib import pyplot as plt

TRAFF_SIGN_NORM_SHAPE = (50, 50)
CANNY_THRESHOLDS = (2, 5)

SAVE_PATH = "../models/signs/digits"

def display_edgemap(edges):
    plt.title("Edgemap")
    plt.imshow(edges, cmap="gray")

    plt.show()

def extract_digits(img):
    resized = cv2.resize(img, TRAFF_SIGN_NORM_SHAPE)
    resized = resized[8:43, 8:43]

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    threshold = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)

    edges = cv2.Canny(threshold, *CANNY_THRESHOLDS)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    #display_edgemap(edges)

    filtered_contours = [cv2.boundingRect(x) for x in cnts if cv2.contourArea(x) > 50.0]
    sorted_contours = sorted(filtered_contours, key=lambda i: i[0])

    digit_frames = []

    for cnt in sorted_contours:
        x, y, w, h = cnt

        digit_frame = threshold[y : y + h, x : x + w]
        digit_frames.append(digit_frame)

        cv2.rectangle(resized, (x, y), (x + w, y + h), (0, 255, 0), 1)

    cv2.imshow("Threshold", threshold)
    cv2.imshow("Image", resized)

    return digit_frames
        
def save_for_training(frame, name):
    digits = "0123456789"
    randomized = "".join(random.sample(digits, len(digits)))

    path = f"{SAVE_PATH}/{name}/{randomized}.png"

    cv2.imwrite(path, frame)

def iterate_dataset():
    project = sly.Project("C:\\Users\\NAM4SF\\Documents\\road-signs", sly.OpenMode.READ)

    for dataset in project.datasets:
        items = dataset.items()

        for name, image_path, ann_path in items:
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            shape = img.shape[:2]

            if shape[0] < 50 or shape[1] < 50:
                continue

            ann_file = open(ann_path)
            ann_json = json.load(ann_file)
            
            ann = sly.Annotation.from_json(ann_json, project.meta)

            for label in ann.labels:
                cl_name = label.obj_class.name

                if cl_name != "speedlimit":
                    continue

                geom = label.geometry.to_json()
                points = geom["points"]["exterior"]

                #cv2.rectangle(img, tuple(points[0]), tuple(points[1]), (0, 255, 0), 1)
                sl_frame = img[points[0][1] : points[1][1], points[0][0] : points[1][0]]

                digits = extract_digits(sl_frame)

                key = cv2.waitKey(0)

                if key == ord("q"):
                    cv2.destroyAllWindows()
                    return
                elif key == ord("e"):
                    cv2.destroyAllWindows()
                    continue
                elif key == ord("s"):
                    inp = input().split(",")
                    for i, j in enumerate(inp):
                        save_for_training(digits[i], j)
                
                cv2.destroyAllWindows()

iterate_dataset()