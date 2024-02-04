#!/bin/bash

CAM_WIDTH=640
CAM_HEIGHT=480
CAM_TIMEOUT=1000

NUM_SAMPLES=6
SAVE_PATH="../samples/calibration/cm4"

setup_vars() {
	read -p "Do you want to take a custom amount of samples? [y/n]: " CUST_AMN_RESP

	if [ "$CUST_AMN_RESP" = "y" ]; then
		read -p "Enter desired amount: " NUM_SAMPLES
	fi

	echo "Final number of samples to be taken: $NUM_SAMPLES"
}

take_samples() {
	for i in $(seq $1 $NUM_SAMPLES)
	do
		read -p "Take snap #$i? [y/n]: " USER_RESP

		if [ "$USER_RESP" = "y" ]; then	
			echo "Keep the object of interest still!"
			rpicam-still --camera 0 -t $CAM_TIMEOUT --width $CAM_WIDTH --height $CAM_HEIGHT -o "${SAVE_PATH}/cam0/calib_$i.jpg"
			rpicam-still --camera 1 -t $CAM_TIMEOUT --width $CAM_WIDTH --height $CAM_HEIGHT -o "${SAVE_PATH}/cam1/calib_$i.jpg"
		fi
	done
}

setup_vars
take_samples $1
