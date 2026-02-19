#include <Servo.h>

Servo myServo;
bool servoActive = false;

void setup() {
  pinMode(29, OUTPUT);    // LED on pin 29
  Serial.begin(9600);     // Serial communication with Python
  myServo.attach(2);      // Servo connected pin 2
  myServo.write(0);       // Initial position
}

void loop() {
  // reading serial input
  if (Serial.available()) {
    char c = Serial.read();

    if (c == '1') {
      digitalWrite(29, HIGH);   // turn on LED
      servoActive = true;       // Activate Servo

    } else if (c == '0') {
      digitalWrite(12, LOW);    // Turn off led
      servoActive = false;      // Stop servo
      myServo.write(0);         // Return to initial position
    }
  }

  // if active run
  if (servoActive) {
    for (int pos = 0; pos <= 90; pos += 2) {
      myServo.write(pos);
      delay(20);
    }

    for (int pos = 90; pos >= 0; pos -= 2) {
      myServo.write(pos);
      delay(20);
    }
  }

  delay(10);  //small pause for smooth loop
}
