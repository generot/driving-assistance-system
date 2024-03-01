import io
import time
import cv2

import multiprocessing as mp

from picamera2 import Picamera2

def camera_init(cam_id):
    cam = Picamera2(cam_id)
    config = cam.create_preview_configuration({ 'size': (1280, 720), 'format': 'YUV420' })

    cam.configure(config)
    cam.start()

    time.sleep(1)

    return cam

def get_frame(cam):
    fr = cam.capture_array()
    fr = cv2.cvtColor(fr, cv2.COLOR_YUV420p2RGB)

    return fr

def get_frames_stereo(cam1, cam2):
    fr1 = cam1.capture_array()
    fr2 = cam2.capture_array()

    #fr1 = cv2.cvtColor(fr1, cv2.COLOR_YUV420p2RGB)
    #fr2 = cv2.cvtColor(fr2, cv2.COLOR_YUV420p2RGB)

    return (fr1, fr2)

def cv_loop(q1, q2):
    while cv2.waitKey(1) != ord('e'):
        fr1, fr2 = q1.get(), q2.get()

        fr1 = cv2.cvtColor(fr1, cv2.COLOR_YUV420p2RGB)
        fr2 = cv2.cvtColor(fr2, cv2.COLOR_YUV420p2RGB)

        cv2.imshow('frame1', fr1)
        cv2.imshow('frame2', fr2)

def main_parallel():
    mp.set_start_method("spawn")

    q1 = mp.Queue()
    q2 = mp.Queue()

    cv_proc = mp.Process(target=cv_loop, args=(q1, q2))
    cv_proc.start()

    cam1 = camera_init(0)
    cam2 = camera_init(1)

    while True:
        fr1_raw, fr2_raw = get_frames_stereo(cam1, cam2)

        q1.put(fr1_raw)
        q2.put(fr2_raw)

def main():
    cam1 = camera_init(0)
    cam2 = camera_init(1)

    while cv2.waitKey(1) != ord('e'):
        #fr1, fr2 = get_frames_stereo(cam1, cam2)
        fr1 = get_frame(cam1)

        cv2.imshow('frame1', fr1)
        #cv2.imshow('frame2', fr2)


if __name__ == "__main__":
    main()

'''
if __name__ == "__main__":
    mp.set_start_method("spawn")

    q1, q2 = mp.Queue(), mp.Queue()

    child = mp.Process(target=pull_frame_loop, args=(q1, q2))
    child.start()

    display_frame_loop(q1, q2)
'''
