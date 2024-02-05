import io
import time
import cv2

from picamera2 import Picamera2

CAM_WIDTH = 640
CAM_HEIGHT = 480

def camera_init(cam_id):
    cam = Picamera2(cam_id)
    config = cam.create_preview_configuration({ 'size': (CAM_WIDTH, CAM_HEIGHT), 'format': 'YUV420' })

    cam.configure(config)
    cam.start()

    time.sleep(1)

    return cam

def get_frame(cam):
    fr = cam.capture_array()
    fr = cv2.cvtColor(fr, cv2.COLOR_YUV420p2RGB)

    return fr

def main():
    cam1 = camera_init(0)
    cam2 = camera_init(1)

    while cv2.waitKey(1) != ord('e'):
        #fr1, fr2 = get_frames_stereo(cam1, cam2)
        fr1 = get_frame(cam1)
        fr2 = get_frame(cam2)

        cv2.imshow('frame1', fr1)
        cv2.imshow('frame2', fr2)


if __name__ == "__main__":
    main()

