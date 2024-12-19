

void setup() {
  Serial.begin(9600, SERIAL_8N1);
  //Serial.setTimeout(1);
}

void loop() {
  /*byte byte_mask = B11111111;
  for (int i = 0; i <= 300; ++i) {
    byte bytes[] = {
      i & byte_mask,
      i & (byte_mask << 8),
      i & (byte_mask << 16),
      i & (byte_mask << 24)
    };
    Serial.write(bytes, 4);
  }*/
  if (Serial.available()) {

    byte buffer[4];
    Serial.readBytes(buffer, 4);
    long value = 0;
    value += buffer[0] << 24;
    value += buffer[1] << 16;
    value += buffer[2] << 8;
    value += buffer[3];
    value += 1;
    
    byte new_buff[4];
    new_buff[3] = (byte) value;
    new_buff[2] = (byte) value >> 8;
    new_buff[1] = (byte) value >> 16;
    new_buff[0] = (byte) value >> 24;
    Serial.write(new_buff, 4);
  }
  delay(100);
}