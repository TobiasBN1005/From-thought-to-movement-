#include <Wire.h>
#include <I2Cdev.h>
#include <MPU6050_6Axis_MotionApps20.h>
#include <Servo.h>

MPU6050 mpu;
Servo servoPitch;  // håndens vinkel op/ned
Servo servoYaw;    // håndens rotation

bool dmpReady = false;
uint8_t mpuIntStatus;
uint8_t devStatus;
uint16_t packetSize;
uint8_t fifoBuffer[64];
Quaternion q;
VectorFloat gravity;
float ypr[3];  // yaw, pitch, roll

void setup() {
  Serial.begin(115200);
  Wire.begin();

  // Initialiser MPU6050
  mpu.initialize();
  devStatus = mpu.dmpInitialize();

  if (devStatus == 0) {
    mpu.setDMPEnabled(true);
    dmpReady = true;
    packetSize = mpu.dmpGetFIFOPacketSize();
  } else {
    Serial.println("DMP init failed");
    while (1);
  }

  // Tilslut servoer
  servoPitch.attach(9);  // vippe servo
  servoYaw.attach(10);   // dreje servo
}

void loop() {
  if (!dmpReady) return;

  if (mpu.dmpGetCurrentFIFOPacket(fifoBuffer)) {
    mpu.dmpGetQuaternion(&q, fifoBuffer);
    mpu.dmpGetGravity(&gravity, &q);
    mpu.dmpGetYawPitchRoll(ypr, &q, &gravity);

    float yaw   = ypr[0] * 180/M_PI;   // grader
    float pitch = ypr[1] * 180/M_PI;

    // Begræns pitch og yaw mellem -90 og 90 grader
    pitch = constrain(pitch, -90, 90);
    yaw   = constrain(yaw, -90, 90);

    // Map værdier til 0–180° for servoer
    int pitchServo = map(pitch, -90, 90, 0, 180);
    int yawServo   = map(yaw, -90, 90, 0, 180);

    servoPitch.write(pitchServo);
    servoYaw.write(yawServo);

    // Debug
    Serial.print("Pitch: "); Serial.print(pitch);
    Serial.print(" | Servo: "); Serial.print(pitchServo);
    Serial.print(" || Yaw: "); Serial.print(yaw);
    Serial.print(" | Servo: "); Serial.println(yawServo);
  }
}