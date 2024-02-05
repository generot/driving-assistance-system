#!../venv/bin/python

import cv2
import numpy as np
import multiprocessing as mp

from rpicam import camera_init, get_frame
from traff_sign import train_knn
from detect import classify_car_rear, average_box_init, speed_limit_rec
from stereo import project_to_3d, match_roi, get_world_dist

from obd_iface import obd_loop
from prph_gpio import periphery_table

FPS = 90
SPEED_INCR = 5

stereo_matrices = np.load("../data/stereo_calib_mats.npz")

def main(v_speed):
    cam0 = camera_init(0)
    cam1 = camera_init(1)

    last_detected_sl = 0
    last_distance = 0
    speed_to_keep = 0

    get_average_box = average_box_init()
    #knn = train_knn()

    def incr_speed(sign):
        nonlocal last_detected_sl
        nonlocal speed_to_keep

        increment = SPEED_INCR * sign

        if last_detected_sl != 0 \
        and v_speed.value + increment > last_detected_sl \
        and v_speed.value + increment <= 0:
            return

        speed_to_keep += increment

    def check_distance(veh_dist, veh_speed):
        if veh_speed == 0:
            return

        dist_m = veh_dist / 100
        speed_meters_ps = (veh_speed * 1000) / 3600

        time = dist_m / speed_meters_ps

        WARN_THRESHOLD = 3 #s
        ALRT_THRESHOLD = 1 #s

        print(time)

        if ALRT_THRESHOLD < time and time < WARN_THRESHOLD:
            periphery_table["led_warn"].on()
        elif time < ALRT_THRESHOLD:
            periphery_table["led_warn"].on()
            periphery_table["led_alert"].on()
            periphery_table["buzzer"].on()
        else:
            periphery_table["led_warn"].off()
            periphery_table["led_alert"].off()
            periphery_table["buzzer"].off()

    def reset_sl():
        nonlocal last_detected_sl
        last_detected_sl = 0

    def increase_speed():
        incr_speed(1)

    def decrease_speed():
        incr_speed(-1)

    periphery_table["but_1"].when_pressed = increase_speed
    periphery_table["but_2"].when_pressed = decrease_speed
    periphery_table["but_3"].when_pressed = reset_sl

    while True:
        frame = get_frame(cam0)
        frame1 = get_frame(cam1)

        frame = cv2.convertScaleAbs(frame, alpha=0.8, beta=10)

        #car_frame, result = classify_car_rear(frame, (0, 480), (100, 380))
        car_frame, result = classify_car_rear(frame)

        avg = get_average_box(result)
        
        if avg[2] != 0 and avg[3] != 0:
            other_upper_left, _, _ = match_roi(frame1, car_frame, avg)
            points_3d = project_to_3d(stereo_matrices, (avg[0], avg[1]), other_upper_left)

            dist = get_world_dist(stereo_matrices, points_3d)

            last_distance = dist

        #last_detected_sl = speed_limit_rec(frame)

        check_distance(last_distance, v_speed.value)
        #check_distance(1000, 50)
        
        '''
        cv2.putText(frame, f"Speed Limit: {last_detected_sl} km / h", 
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        '''
        cv2.putText(frame, f"Distance from front vehicle: {last_distance} cm", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)

        cv2.putText(frame, f"Speed to keep: {speed_to_keep} km/h", 
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 1)
 
        cv2.rectangle(car_frame, (avg[0], avg[1]), (avg[0] + avg[2], avg[1] + avg[3]), (0, 255, 0))

        cv2.imshow("Camera 1", frame)
        cv2.imshow("Camera 2", frame1)
        #cv2.imshow("Only Red", car_frame)

        if cv2.waitKey(1000 // FPS) == ord('e'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    vehicle_speed = mp.Value("i", 0)

    obd_p = mp.Process(target=obd_loop, args=(vehicle_speed,))
    #obd_p.start()
    #obd_p.join()

    main(vehicle_speed)
