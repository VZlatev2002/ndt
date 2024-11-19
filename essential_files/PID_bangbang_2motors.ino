#include <SimplyAtomic.h>

// Motor 1
#define ENCA1 0
#define ENCB1 1
#define PWM_pin1 2
#define IN1 3
#define IN2 4

// Motor 2
#define ENCA2 6
#define ENCB2 7
#define PWM_pin2 8
#define IN3 9
#define IN4 10

// Motor 1
volatile int posi1 = 0;
float ref1 = 0;
int pos1 = 0;
int e1 = 0;
int u1 = 0;
float integral1 = 0;
float prev_error1 = 0;
float derivative1 = 0;

// Motor 2
volatile int posi2 = 0;
float ref2 = 0;
int pos2 = 0;
int e2 = 0;
int u2 = 0;
float integral2 = 0;
float prev_error2 = 0;
float derivative2 = 0;

// Common control variables
int print_interval = 100;
int interval_count = 0;
int u_max = 255;  // Maximum PWM value
float Kp = 1.75;  // Proportional gain
float Ki = 0.01;  // Integral gain - adjust this value as needed
float Kd = 0.1;   // Derivative gain - adjust this value as needed
int bang_bang_threshold = 1000;  // Threshold for switching to bang-bang control
int bang_bang_speed = 200;  // PWM value for bang-bang control
float counts_per_rotation = 131.25 * 16;
float ref_amplitude = counts_per_rotation;
int rotations = 0;
float time_per_rotation = 5000;
long time_start = 0;

void setup() {
  Serial.begin(230400);

  // Motor 1
  pinMode(ENCA1, INPUT);
  pinMode(ENCB1, INPUT);
  attachInterrupt(digitalPinToInterrupt(ENCA1), readEncoder1, RISING);

  pinMode(PWM_pin1, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);

  // Motor 2
  pinMode(ENCA2, INPUT);
  pinMode(ENCB2, INPUT);
  attachInterrupt(digitalPinToInterrupt(ENCA2), readEncoder2, RISING);

  pinMode(PWM_pin2, OUTPUT);  // Corrected to PWM_pin2
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);

  time_start = millis();
}

void loop() {
  if ((millis() - time_start) > time_per_rotation) {
    rotations++;
    time_start = millis();
  }

  ref1 = ref_amplitude * rotations;
  ref2 = ref_amplitude * rotations;

  ATOMIC() {
    pos1 = posi1;
    pos2 = posi2;
  }

  // Motor 1 PID control
  e1 = ref1 - pos1;

  if (abs(e1) > bang_bang_threshold) {
    // Bang-bang control for large errors
    u1 = (e1 > 0) ? bang_bang_speed : -bang_bang_speed;
  } else {
    // PID control for small errors
    integral1 += e1;  // Accumulate the integral
    derivative1 = e1 - prev_error1;  // Calculate the derivative
    u1 = (Kp * e1) + (Ki * integral1) + (Kd * derivative1);  // PID formula

    // Limit control signal to maximum PWM value
    u1 = constrain(u1, -u_max, u_max);
  }
  prev_error1 = e1;  // Store the previous error for the next loop

  // Determine direction and PWM value
  int dir1 = (u1 >= 0) ? 1 : -1;
  int pwmVal1 = abs(u1);

  // Drive motor 1
  setMotor(dir1, pwmVal1, PWM_pin1, IN1, IN2);

  // Motor 2 PID control
  e2 = ref2 - pos2;

  if (abs(e2) > bang_bang_threshold) {
    // Bang-bang control for large errors
    u2 = (e2 > 0) ? bang_bang_speed : -bang_bang_speed;
  } else {
    // PID control for small errors
    integral2 += e2;  // Accumulate the integral
    derivative2 = e2 - prev_error2;  // Calculate the derivative
    u2 = (Kp * e2) + (Ki * integral2) + (Kd * derivative2);  // PID formula

    // Limit control signal to maximum PWM value
    u2 = constrain(u2, -u_max, u_max);
  }
  prev_error2 = e2;  // Store the previous error for the next loop

  // Determine direction and PWM value
  int dir2 = (u2 >= 0) ? 1 : -1;
  int pwmVal2 = abs(u2);

  // Drive motor 2
  setMotor(dir2, pwmVal2, PWM_pin2, IN3, IN4);

  // Print debug information
  interval_count++;
  if (interval_count >= print_interval) {
    interval_count = 0;
    Serial.print("Motor 1: ");
    Serial.print(ref1);
    Serial.print(" ");
    Serial.print(pos1);
    Serial.print(" ");
    Serial.print(e1);
    Serial.print(" ");
    Serial.print(u1);
    Serial.print("\t");

    Serial.print("Motor 2: ");
    Serial.print(ref2);
    Serial.print(" ");
    Serial.print(pos2);
    Serial.print(" ");
    Serial.print(e2);
    Serial.print(" ");
    Serial.print(u2);
    Serial.println();
  }
}

void setMotor(int dir, float pwmVal, int pwm_pin, int in1, int in2) {
  analogWrite(pwm_pin, pwmVal);
  if (dir == 1) {
    digitalWrite(in1, HIGH);
    digitalWrite(in2, LOW);
  } else if (dir == -1) {
    digitalWrite(in1, LOW);
    digitalWrite(in2, HIGH);
  } else {
    digitalWrite(in1, LOW);
    digitalWrite(in2, LOW);
  }
}

void readEncoder1() {
  int b = digitalRead(ENCB1);
  if (b > 0) {
    posi1++;
  } else {
    posi1--;
  }
}

void readEncoder2() {
  int b = digitalRead(ENCB2);
  if (b > 0) {
    posi2++;
  } else {
    posi2--;
  }
}
