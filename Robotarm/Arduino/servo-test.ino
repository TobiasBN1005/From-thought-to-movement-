#include <Servo.h>

Servo myServo;  // Opretter et servo objekt

//servo1: pin 12 ✅ (grænse: +x til +180 )
//servo2: pin 10 ✅ (0 = 20 deg venstre  over sort, tape ned. Grænse: 0 til 180)
//servo3: pin 8 🚧 (virker, men laver små ryk)
//servo4: pin 6 ✅ (virker, dog tapet fast)
//servo5: pin 4 🚧 (bevæger sig, men hopper i sin yderposition)
//servo6: pin 2 ✅ (grænse: 0 til  +120)
//sæt gerne din max lidt lavere end grænsen
void setup() {
  
    myServo.attach(10);
    for (int pos = 0; pos <= 180 ; pos += 5) {  // Bevæg langsomt til midterposition
        myServo.write(pos);
        delay(100);
    }
}


void loop() {
    

    delay(500); // Pause i startposition
}