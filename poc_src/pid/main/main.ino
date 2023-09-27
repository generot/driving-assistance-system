#include <Servo.h>
#include <math.h>

#define BAUD_RATE 9600

#define PI 3.1415926

#define SERVO_PIN 6
#define SERVO_TORQUE 25 //N-cm
#define FULL_SERVO_REV 180

#define SIM_THROTTLE_MAX_FORCE 36 //N
#define SIM_FULL_SERVO_SWING 1 //cm

#define SIM_VEHICLE_MAX_REVS 4000 // RPM
#define SIM_VEHICLE_WHEEL_DIAM 60 //cm

#define RAD(x) ((x) * PI / 180.0f)

enum Gear {
  FIRST_GEAR = 10,
  SECOND_GEAR = 5,
  THIRD_GEAR = 2,
  FOURTH_GEAR = 1
};

Servo servo;

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

float get_servo_swing() {
  return sin(RAD(angle)) * SIM_FULL_SERVO_SWING;
}

float get_servo_actual_torque(float swing) {
  return SERVO_TORQUE / swing;
}

void rotate_servo(int ang) {
  angle += ang;
  servo.write(angle);
}

void setup() {
  pinMode(SERVO_PIN, OUTPUT);

  Serial.begin(BAUD_RATE);

  float curr_speed = accelerate(press_throttle(10.0f), SECOND_GEAR);
  Serial.println(curr_speed);

  servo.attach(SERVO_PIN);
  servo.write(0);

  delay(2000);
  rotate_servo(20);
  delay(2000);
  rotate_servo(20);
}

void loop() {
}
