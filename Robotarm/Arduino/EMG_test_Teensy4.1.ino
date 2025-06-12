// blå elektrode: aktiv
// rød elektrode: reference
// sort elektrode: ground
// rød ledning: power
// sort ledning: ground
// hvid ledning: Raw EMG signal
// gul ledning: Filtertet/enveloped EMG signal

const int EMG_PIN = A14;  // Gul ledning (filtered output)

void setup() {
  Serial.begin(115200);           // Hurtigere dataoverførsel
  analogReadResolution(12);       // 12-bit opløsning (0–4095)
}

void loop() {
  int emgRawValue = analogRead(EMG_PIN);
  Serial.println(emgRawValue);
  delay(100);  // ~500 Hz sampling, egnet til EMG envelope
}
