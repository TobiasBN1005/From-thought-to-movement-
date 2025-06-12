#include <Wire.h>
#include "MPU6050_TBN.h"  // My own MPU6050 library

const int ledPin = 29;
const int EMG_PIN = A14;

MPU6050_TBN imu1(0x69, &Wire);    // IMU1 adress on 0x69 via Wire
MPU6050_TBN imu2(0x69, &Wire1);   // IMU2 adress on 0x69 via Wire1
MPU6050_TBN imu3(0x69, &Wire2);   // IMU3 adress on 0x69 via Wire2

unsigned long lastEMGTime = 0;
unsigned long lastIMUTime = 0;
const unsigned long emgInterval = 4;   // 250 Hz
const unsigned long imuInterval = 4;  // 250 Hz

void setup() {
  pinMode(ledPin, OUTPUT);
  Serial.begin(115200);
  analogReadResolution(12);
  while (!Serial);  // Wait for USB

  // === I2C initialisation ===
  Wire.begin();     // IMU1
  Wire1.begin();    // IMU2
  Wire2.begin();    // IMU3

  imu1.initialize();
  imu2.initialize();
  imu3.initialize();

  // === Test forbindelser ===
  if (imu1.testConnection()) Serial.println("✅ IMU1 Connected");
  else Serial.println("❌ IMU1 not connected");

  if (imu2.testConnection()) Serial.println("✅ IMU2 connected");
  else Serial.println("❌ IMU2 not connected");

  if (imu3.testConnection()) Serial.println("✅ IMU3 connected");
  else Serial.println("❌ IMU3 not connected");

  Serial.println("✅ Teensy ready");
  digitalWrite(ledPin, HIGH); delay(1000); digitalWrite(ledPin, LOW);
}

void loop() {
  // === LED control via serial ===
  if (Serial.available()) {
    char c = Serial.read();
    if (c == '0') { digitalWrite(ledPin, HIGH); Serial.println("🔵 Recieved 0 → LED TURNED ON"); }
    else if (c == '1') { digitalWrite(ledPin, LOW); Serial.println("⚪ Recieved 1 → LED TURNED OFF"); }
    else { Serial.print("❓ Unknown command: "); Serial.println(c); }
  }

  // === EMG-measureing ===
  unsigned long now = millis();
  if (now - lastEMGTime >= emgInterval) {
    lastEMGTime = now;
    int emgValue = analogRead(EMG_PIN);
    Serial.print("EMG:"); Serial.println(emgValue);
  }

  // === IMU-measurements ===
  if (now - lastIMUTime >= imuInterval) {
    lastIMUTime = now;

    int16_t ax, ay, az, gx, gy, gz;

    // IMU1
  int16_t ax1, ay1, az1, gx1, gy1, gz1;
  imu1.getMotion6(&ax1, &ay1, &az1, &gx1, &gy1, &gz1);
  Serial.print("IMU1:"); Serial.print(ax1); Serial.print(","); Serial.print(ay1); Serial.print(","); Serial.print(az1); Serial.print(",");
  Serial.print(gx1); Serial.print(","); Serial.print(gy1); Serial.print(","); Serial.println(gz1);

  // IMU2
  int16_t ax2, ay2, az2, gx2, gy2, gz2;
  imu2.getMotion6(&ax2, &ay2, &az2, &gx2, &gy2, &gz2);
  Serial.print("IMU2:"); Serial.print(ax2); Serial.print(","); Serial.print(ay2); Serial.print(","); Serial.print(az2); Serial.print(",");
  Serial.print(gx2); Serial.print(","); Serial.print(gy2); Serial.print(","); Serial.println(gz2);

  // IMU3
  int16_t ax3, ay3, az3, gx3, gy3, gz3;
  imu3.getMotion6(&ax3, &ay3, &az3, &gx3, &gy3, &gz3);
  Serial.print("IMU3:");
  Serial.print(ax3); Serial.print(",");
  Serial.print(ay3); Serial.print(",");
  Serial.print(az3); Serial.print(",");
  Serial.print(gx3); Serial.print(",");
  Serial.print(gy3); Serial.print(",");
  Serial.println(gz3);


  }
}
