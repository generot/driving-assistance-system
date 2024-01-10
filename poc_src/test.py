import cv2

#cam = cv2.VideoCapture(0)
cam = cv2.VideoCapture(0)

FPS = 30

while cam.isOpened():
    ret, frame = cam.read()

    if ret == True:
        break

    cv2.imshow('frame', frame)

    if cv2.waitKey(1000 // FPS) == ord('e'):
        break

cam.release()

cv2.destroyAllWindows()
