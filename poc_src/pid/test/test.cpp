#include <iostream>
#include <cmath>
#include <chrono>

#define BAUD_RATE 9600

#define PI 3.1415926

#define SERVO_PIN 6
#define SERVO_TORQUE 25 //N-cm
#define FULL_SERVO_REV 180

#define SIM_THROTTLE_MAX_FORCE 36 //N
#define SIM_FULL_SERVO_SWING 1.0f //cm

#define SIM_VEHICLE_MAX_REVS 4000 // RPM
#define SIM_VEHICLE_WHEEL_DIAM 60 //cm

#define RAD(x) ((x) * PI / 180.0f)

using std::cout;

enum Gear {
  FIRST_GEAR = 10,
  SECOND_GEAR = 5,
  THIRD_GEAR = 2,
  FOURTH_GEAR = 1
};

struct pid_coeff {
	float Kp;
	float Ki;
	float Kd;
};

int angle = 0;
float sim_vehicle_speed = 0.0f;

// In this setup, the acceleration is immedeate - once the throttle is pressed with a certain force,
// the engine immedeately revolves with the appropriate RPM, and the transmission immedeately revolves
// with the proportionally corect RPM.
float accelerate(float rpm, float gear_ratio) {
  float transmission_rpm = rpm / gear_ratio;
  float vehicle_speed = transmission_rpm * SIM_VEHICLE_WHEEL_DIAM * PI; //cm / min

  return vehicle_speed * (6.0f / 10000.0f);
}

float press_throttle(float force) {
  return (float)SIM_VEHICLE_MAX_REVS * force / (float)SIM_THROTTLE_MAX_FORCE;
}

float get_servo_swing(float extension = 0.0f) {
  return sin(RAD(angle)) * (SIM_FULL_SERVO_SWING + extension);
}

float get_servo_force(float swing) {
  return SERVO_TORQUE / swing;
}

void rotate_servo(int ang) {
  angle += ang;
}

float pid_ctrl(float sp, float ep, pid_coeff *coeffs) {
	
}

int main() {
	auto begin = std::chrono::high_resolution_clock::now();

	float current_speed = 0.0f;
	float desired_speed = 40.0f;

	pid_coeff cfs = { 4.0f, 6.0f, 0.0f };

	float p, i, d;
	float last_error = 0.0f;

	p = 0.0f;
	i = 0.0f;
	d = 0.0f;

	float error = desired_speed - current_speed;

	for(int i = 0; i < 10; i++) {
		/*std::chrono::duration<double> elapsed_time = 
			(std::chrono::high_resolution_clock::now() - begin);*/

		p = cfs.Kp * error;
		i += error;
		d = cfs.Kd * (last_error - error) / 2;

		float u = p + cfs.Ki + d;

		current_speed += u;

		cout << current_speed << '\n';

		last_error = error;
	}
	
	return 0;
}
