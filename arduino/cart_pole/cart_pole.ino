int cart_rotary_encoder_identifier = 0;
int cart_rotary_white_pin = 2;
int cart_rotary_green_pin = 4;
volatile long cart_rotary_index = 0;
volatile unsigned long cart_rotary_num_changes = 0;
bool cart_rotary_clockwise = false;

int pole_rotary_encoder_identifier = 1;
int pole_rotary_white_pin = 3;
int pole_rotary_green_pin = 5;
volatile long pole_rotary_index = 0;
volatile unsigned long pole_rotary_num_changes = 0;
bool pole_rotary_clockwise = false;

byte START = 0;
byte GET_STATE = 1;
byte STOP = 2;

void setup() {

  pinMode(cart_rotary_white_pin, INPUT_PULLUP);
  digitalWrite(cart_rotary_white_pin, HIGH);
  pinMode(cart_rotary_green_pin, INPUT_PULLUP);
  digitalWrite(cart_rotary_green_pin, HIGH);

  pinMode(pole_rotary_white_pin, INPUT_PULLUP);
  digitalWrite(pole_rotary_white_pin, HIGH);
  pinMode(pole_rotary_green_pin, INPUT_PULLUP);
  digitalWrite(pole_rotary_green_pin, HIGH);

  Serial.begin(9600, SERIAL_8N1);

}

void cart_white_changed() {
  bool cart_white_pin_value = digitalRead(cart_rotary_white_pin);
  bool cart_rotary_green_value = digitalRead(cart_rotary_green_pin);
  if (cart_white_pin_value == cart_rotary_green_value) {
    cart_rotary_index -= 1;
    cart_rotary_clockwise = false;
  } else {
    cart_rotary_index += 1;
    cart_rotary_clockwise = true;
  }
  cart_rotary_num_changes += 1;
}

void pole_white_changed() {
  bool pole_white_pin_value = digitalRead(pole_rotary_white_pin);
  bool pole_rotary_green_value = digitalRead(pole_rotary_green_pin);
  if (pole_white_pin_value == pole_rotary_green_value) {
    pole_rotary_index -= 1;
    pole_rotary_clockwise = false;
  } else {
    pole_rotary_index += 1;
    pole_rotary_clockwise = true;
  }
  pole_rotary_num_changes += 1;
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
  bytes[3] = (byte)value;
  bytes[2] = (byte)(value >> 8);
  bytes[1] = (byte)(value >> 16);
  bytes[0] = (byte)(value >> 24);
}

void write_long(long value) {
  byte bytes[4];
  long_to_bytes(value, bytes);
  Serial.write(bytes, 4);
}

void write_bool(bool value) {
  Serial.write(value);
}

void loop() {

  if (Serial.available()) {

    // read command
    int command_bytes_len = 2;
    byte command_bytes[command_bytes_len];
    Serial.readBytes(command_bytes, command_bytes_len);
    byte command = command_bytes[0];
    byte identifier = command_bytes[1];

    if (command == START) {
      attachInterrupt(digitalPinToInterrupt(cart_rotary_white_pin), cart_white_changed, CHANGE);
      attachInterrupt(digitalPinToInterrupt(pole_rotary_white_pin), pole_white_changed, CHANGE);
      delay(1000);
      cart_rotary_index = 0;
      pole_rotary_index = 0;
    }

    else if (command == GET_STATE) {

      if (identifier == cart_rotary_encoder_identifier) {
        write_long(cart_rotary_num_changes);
        write_long(cart_rotary_index);
        write_bool(cart_rotary_clockwise);
      }
      else if (identifier == pole_rotary_encoder_identifier) {
        write_long(pole_rotary_num_changes);
        write_long(pole_rotary_index);
        write_bool(pole_rotary_clockwise);
      }
    }

    else if (command == STOP) {
      detachInterrupt(digitalPinToInterrupt(cart_rotary_white_pin));
      detachInterrupt(digitalPinToInterrupt(pole_rotary_white_pin));
    }
  }
}
