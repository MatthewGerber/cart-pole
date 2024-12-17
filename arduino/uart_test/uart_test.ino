

void setup() {
  Serial.begin(9600);
}

void loop() {
  byte byte_mask = B11111111;
  for (int i = 0; i <= 300; ++i) {
    byte bytes[] = {
      i & byte_mask,
      i & (byte_mask << 8),
      i & (byte_mask << 16),
      i & (byte_mask << 24)
    };
    Serial.write(bytes, 4);
  }
}