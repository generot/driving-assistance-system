import io
import time
import cv2

from picamera2 import Picamera2

cameras = []

def stereo_init():
    camera_init(0)
    camera_init(1)

def camera_init(cam_id):
    cam = Picamera2(cam_id)
    config = cam.create_preview_configuration({ 'format': 'RGB888' })

    cam.configure(config)
    cam.start()

    time.sleep(1)
    cameras.append(cam)

def get_frames_stereo():
    fr1 = cameras[0].capture_array()
    fr2 = cameras[1].capture_array()

    return (fr1, fr2)

def main():
    stereo_init()

    while cv2.waitKey(1) != ord('e'):
        fr1, fr2 = get_frames_stereo()

        cv2.imshow('frame1', fr1)
        cv2.imshow('frame2', fr2)

main()
