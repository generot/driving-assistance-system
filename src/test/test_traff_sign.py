import cv2
from impl import traff_sign as tsr

def init_image_manipulation(frame):
    window_name = "Parameter manipulation"

    def run_and_show(frame):
        augmented, dilated, circles = tsr.recognize_sl_sign(frame)

        if isinstance(circles, list):
            for circle in circles[0]:
                x, y, r = circle
                x_int = int(x)
                y_int = int(y)
                r_int = int(r)

                cv2.circle(augmented, (x_int, y_int), r_int, (0, 255, 0), 2)

        cv2.imshow("Real image", augmented)
        cv2.imshow(window_name, dilated)

    def on_change_blur_kernel(val):
        tsr.sign_gauss_blur_kernel = (2 * val + 1, 2 * val + 1)
        run_and_show(frame)

    def on_change_dilation_kernel(val):
        tsr.sign_dilation_kernel = (2 * val + 1, 2 * val + 1)
        run_and_show(frame)

    def on_change_p1(val):
        tsr.sign_hcircles_p1 = val
        run_and_show(frame)

    def on_change_p2(val):
        tsr.sign_hcircles_p2 = val
        run_and_show(frame)
    
    def on_change_max_r(val):
        tsr.sign_hcircles_max_r = val
        run_and_show(frame)

    def on_change_min_r(val):
        tsr.sign_hcircles_min_r = val
        run_and_show(frame)

    run_and_show(frame)

    cv2.createTrackbar("Gaussian blur kernel size", window_name, 7, 20, on_change_blur_kernel)
    cv2.createTrackbar("Dilation kernel size", window_name, 5, 20, on_change_dilation_kernel)
    cv2.createTrackbar("Hough circles P1", window_name, 200, 330, on_change_p1)
    cv2.createTrackbar("Hough circles P2", window_name, 80, 330, on_change_p2)
    cv2.createTrackbar("Max radius", window_name, 40, 80, on_change_max_r)
    cv2.createTrackbar("Max radius", window_name, 40, 80, on_change_min_r)

    cv2.waitKey(0)
    cv2.destroyAllWindows()