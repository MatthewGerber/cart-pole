int cart_rotary_white_pin = 3;
int cart_rotary_green_pin = 5;
volatile long cart_rotary_index = 0;
bool cart_rotary_clockwise = false;

int pole_rotary_white_pin = 2;
int pole_rotary_green_pin = 4;
volatile long pole_rotary_index = 0;
bool pole_rotary_clockwise = false;

void setup() {

  Serial.begin(9600, SERIAL_8N1);

  pinMode(cart_rotary_white_pin, INPUT_PULLUP);
  digitalWrite(cart_rotary_white_pin, HIGH);
  pinMode(cart_rotary_green_pin, INPUT_PULLUP);
  digitalWrite(cart_rotary_green_pin, HIGH);
  attachInterrupt(digitalPinToInterrupt(cart_rotary_white_pin), cart_white_changed, CHANGE);
  
  pinMode(pole_rotary_white_pin, INPUT_PULLUP);
  digitalWrite(pole_rotary_white_pin, HIGH);
  pinMode(pole_rotary_green_pin, INPUT_PULLUP);
  digitalWrite(pole_rotary_green_pin, HIGH);
  attachInterrupt(digitalPinToInterrupt(pole_rotary_white_pin), pole_white_changed, CHANGE);

  delay(1000);

  cart_rotary_index = 0;
  pole_rotary_index = 0;

}

void cart_white_changed() {
  bool cart_white_pin_value = digitalRead(cart_rotary_white_pin);
  bool cart_rotary_green_value = digitalRead(cart_rotary_green_pin);
  if (cart_white_pin_value == cart_rotary_green_value) {
    cart_rotary_index -= 1;
    cart_rotary_clockwise = false;
  }
  else {
    cart_rotary_index += 1;
    cart_rotary_clockwise = true;
  }
}

void pole_white_changed() {
  bool pole_white_pin_value = digitalRead(pole_rotary_white_pin);
  bool pole_rotary_green_value = digitalRead(pole_rotary_green_pin);
  if (pole_white_pin_value == pole_rotary_green_value) {
    pole_rotary_index -= 1;
    pole_rotary_clockwise = false;
  }
  else {
    pole_rotary_index += 1;
    pole_rotary_clockwise = true;
  }
}

long bytes_to_long(byte bytes[]) {
  long value = 0;
  value += ((long)bytes[0]) << 24;
  value += ((long)bytes[1]) << 16;
  value += ((long)bytes[2]) << 8;
  value += ((long)bytes[3]);
  return value;
}

void long_to_bytes(long value, byte bytes[]) {
  bytes[3] = (byte) value;
  bytes[2] = (byte) (value >> 8);
  bytes[1] = (byte) (value >> 16);
  bytes[0] = (byte) (value >> 24);
}

void loop() {
  if (Serial.available()) {

    // read command
    int command_bytes_len = 4;
    byte command_buffer[command_bytes_len];
    Serial.readBytes(command_buffer, command_bytes_len);
    long command = bytes_to_long(command_buffer);

    // write state
    byte cart_rotary_index_bytes[4];
    long_to_bytes(cart_rotary_index, cart_rotary_index_bytes);
    Serial.write(cart_rotary_index_bytes, 4);

    byte pole_rotary_index_bytes[4];
    long_to_bytes(pole_rotary_index, pole_rotary_index_bytes);
    Serial.write(pole_rotary_index_bytes, 4);

  }
}
