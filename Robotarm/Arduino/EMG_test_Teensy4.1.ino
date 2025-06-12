// blue electrode: active electrode
// red electrode: reference
// black electrode: ground
// red wire: power
// black wire: ground
// white wite: Raw EMG signal
// yellow ledning: Enveloped EMG signal

const int EMG_PIN = A14;  // yellow wire (filtered output)

void setup() {
  Serial.begin(115200);           // fast datatransfer
  analogReadResolution(12);       // 12-bit resolution (0–4095)
}

void loop() {
  int emgRawValue = analogRead(EMG_PIN);
  Serial.println(emgRawValue);
  delay(100); 
}
