#include <Servo.h>

Servo myServo;
bool servoActive = false;

void setup() {
  pinMode(29, OUTPUT);    // LED på pin 29
  Serial.begin(9600);     // Serial til kommunikation med Python
  myServo.attach(2);      // Servo tilsluttet pin 2
  myServo.write(0);       // Startposition
}

void loop() {
  // === Læs serielt input ===
  if (Serial.available()) {
    char c = Serial.read();

    if (c == '1') {
      digitalWrite(29, HIGH);   // Tænd LED
      servoActive = true;       // Aktiver servo

    } else if (c == '0') {
      digitalWrite(12, LOW);    // Sluk LED
      servoActive = false;      // Stop servo
      myServo.write(0);         // Returnér til startposition (valgfrit)
    }
  }

  // === Kør servo hvis aktiv ===
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

  delay(10);  // Let pause for stabil loop
}
