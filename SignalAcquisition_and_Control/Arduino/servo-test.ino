#include <Servo.h>

Servo myServo;  // Creates a servo objet

//servo1: pin 12 ✅ (limit: +x to +180 )
//servo2: pin 10 ✅ (0 = 20 deg left  above black, tape down. limit: 0 til 180)
//servo3: pin 8 🚧 (works, but a few twitches)
//servo4: pin 6 ✅ (works)
//servo5: pin 4 🚧 (works, but jumps to its outer position)
//servo6: pin 2 ✅ (limit: 0 to  +120)
//your max should be a tiny bit lower than the limit, to prevent the servos getting stuck in their outer positions
void setup() {
  
    myServo.attach(10);
    for (int pos = 0; pos <= 180 ; pos += 5) {  // Slow movement to the middle
        myServo.write(pos);
        delay(100);
    }
}


void loop() {
    

    delay(500); // Pause in start position
}
