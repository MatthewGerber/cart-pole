

void setup() {
  Serial.begin(115200, SERIAL_8N1);
}

long bytes_to_long(byte bytes[]) {
  long value = 0;
  value += bytes[0] << 24;
  value += bytes[1] << 16;
  value += bytes[2] << 8;
  value += bytes[3];
  return value;
}

byte* long_to_bytes(long value) {
  byte* bytes = new byte[4];
  bytes[3] = (byte) value;
  bytes[2] = (byte) value >> 8;
  bytes[1] = (byte) value >> 16;
  bytes[0] = (byte) value >> 24;
  return bytes;
}

void loop() {
  if (Serial.available()) {
    byte buffer[4];
    Serial.readBytes(buffer, 4);
    long value = bytes_to_long(buffer) + 1;
    byte* return_buffer = long_to_bytes(value);
    Serial.write(return_buffer, 4);
    delete[] return_buffer;
  }
}