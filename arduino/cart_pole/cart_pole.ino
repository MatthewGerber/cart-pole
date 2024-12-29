const size_t floatunion_len = 4;
typedef union
{
  float number;
  uint8_t bytes[floatunion_len];
} FLOATUNION_t;

int cart_rotary_encoder_identifier = 0;
int cart_rotary_white_pin;
int cart_rotary_green_pin;
volatile long cart_rotary_index;
volatile bool cart_rotary_clockwise = false;
int cart_rotary_phase_changes_per_rotation;
float cart_rotary_phase_changes_per_degree;
FLOATUNION_t cart_rotary_net_degrees;
FLOATUNION_t cart_velocity;
FLOATUNION_t cart_velocity_step_size;
FLOATUNION_t cart_acceleration;
FLOATUNION_t cart_acceleration_step_size;
int cart_state_update_hz;
float cart_state_update_interval_ms;
unsigned long cart_rotary_state_time_ms;

int pole_rotary_encoder_identifier = 1;
int pole_rotary_white_pin;
int pole_rotary_green_pin;
volatile long pole_rotary_index;
volatile bool pole_rotary_clockwise = false;
int pole_rotary_phase_changes_per_rotation;
float pole_rotary_phase_changes_per_degree;
FLOATUNION_t pole_rotary_net_degrees;
FLOATUNION_t pole_velocity;
FLOATUNION_t pole_velocity_step_size;
FLOATUNION_t pole_acceleration;
FLOATUNION_t pole_acceleration_step_size;
int pole_state_update_hz;
float pole_state_update_interval_ms;
unsigned long pole_rotary_state_time_ms;

byte START = 1;
byte GET_STATE = 2;
byte WAIT_FOR_STATIONARITY = 3;
byte STOP = 4;

void setup() {

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
  size_t len = 4;
  byte bytes[len];
  long_to_bytes(value, bytes);
  Serial.write(bytes, len);
}

void write_float(FLOATUNION_t f) {
  Serial.write(f.bytes, floatunion_len);
}

void write_bool(bool value) {
  Serial.write(value);
}

void loop() {

  unsigned long curr_time_ms = millis();

  if (cart_rotary_state_time_ms == -1 || curr_time_ms - cart_rotary_state_time_ms >= cart_state_update_interval_ms) {

  }

  unsigned long pole_rotary_state_elapsed_ms = curr_time_ms - pole_rotary_state_time_ms;
  if (pole_rotary_state_elapsed_ms >= pole_state_update_interval_ms) {
    float net_total_degrees = pole_rotary_index / pole_rotary_phase_changes_per_degree;
    float elapsed_seconds = pole_rotary_state_elapsed_ms / 1000.0;
    float previous_pole_velocity = pole_velocity.number;
    float current_pole_velocity = (net_total_degrees - pole_rotary_net_degrees.number) / elapsed_seconds;
    pole_velocity.number = (1.0 - pole_velocity_step_size.number) * previous_pole_velocity + pole_velocity_step_size.number * current_pole_velocity;
    float current_pole_acceleration = (current_pole_velocity - previous_pole_velocity) / elapsed_seconds;
    pole_acceleration.number = (1.0 - pole_acceleration_step_size.number) * pole_acceleration.number + pole_acceleration_step_size.number * current_pole_acceleration;
    pole_rotary_state_time_ms = curr_time_ms;
    pole_rotary_net_degrees.number = net_total_degrees;
  }

  if (Serial.available()) {

    // read command
    size_t command_bytes_len = 2;
    byte command_bytes[command_bytes_len];
    Serial.readBytes(command_bytes, command_bytes_len);
    byte command = command_bytes[0];
    byte identifier = command_bytes[1];

    if (command == START) {

      size_t start_bytes_len = 14;
      byte start_bytes[start_bytes_len];
      Serial.readBytes(start_bytes, start_bytes_len);

      if (identifier == cart_rotary_encoder_identifier) {
        cart_rotary_white_pin = start_bytes[0];
        pinMode(cart_rotary_white_pin, INPUT_PULLUP);
        digitalWrite(cart_rotary_white_pin, HIGH);
        cart_rotary_green_pin = start_bytes[1];
        pinMode(cart_rotary_green_pin, INPUT_PULLUP);
        digitalWrite(cart_rotary_green_pin, HIGH);

        // todo:  2 bytes for phase changes per rotation
        cart_rotary_phase_changes_per_rotation = 1200;
        cart_rotary_phase_changes_per_degree = cart_rotary_phase_changes_per_rotation / 360.0;

        // todo:  1 byte for phase-change mode

        cart_velocity_step_size.bytes[0] = start_bytes[5];
        cart_velocity_step_size.bytes[1] = start_bytes[6];
        cart_velocity_step_size.bytes[2] = start_bytes[7];
        cart_velocity_step_size.bytes[3] = start_bytes[8];

        cart_acceleration_step_size.bytes[0] = start_bytes[9];
        cart_acceleration_step_size.bytes[1] = start_bytes[10];
        cart_acceleration_step_size.bytes[2] = start_bytes[11];
        cart_acceleration_step_size.bytes[3] = start_bytes[12];

        cart_state_update_hz = start_bytes[13];
        cart_state_update_interval_ms = 1000.0 / (float)cart_state_update_hz;

        attachInterrupt(digitalPinToInterrupt(cart_rotary_white_pin), cart_white_changed, CHANGE);
        delay(1000);
        cart_rotary_index = 0;
        cart_rotary_net_degrees.number = 0.0;
        cart_velocity.number = 0.0;
        cart_acceleration.number = 0.0;
        cart_rotary_state_time_ms = millis();
      }
      else if (identifier == pole_rotary_encoder_identifier) {
        pole_rotary_white_pin = start_bytes[0];
        pinMode(pole_rotary_white_pin, INPUT_PULLUP);
        digitalWrite(pole_rotary_white_pin, HIGH);
        pole_rotary_green_pin = start_bytes[1];
        pinMode(pole_rotary_green_pin, INPUT_PULLUP);
        digitalWrite(pole_rotary_green_pin, HIGH);

        // todo:  2 bytes for phase changes per rotation
        pole_rotary_phase_changes_per_rotation = 1200;
        pole_rotary_phase_changes_per_degree = pole_rotary_phase_changes_per_rotation / 360.0;

        // todo:  1 byte for phase-change mode

        pole_velocity_step_size.bytes[0] = start_bytes[5];
        pole_velocity_step_size.bytes[1] = start_bytes[6];
        pole_velocity_step_size.bytes[2] = start_bytes[7];
        pole_velocity_step_size.bytes[3] = start_bytes[8];

        pole_acceleration_step_size.bytes[0] = start_bytes[9];
        pole_acceleration_step_size.bytes[1] = start_bytes[10];
        pole_acceleration_step_size.bytes[2] = start_bytes[11];
        pole_acceleration_step_size.bytes[3] = start_bytes[12];

        pole_state_update_hz = start_bytes[13];
        pole_state_update_interval_ms = 1000.0 / (float)pole_state_update_hz;

        attachInterrupt(digitalPinToInterrupt(pole_rotary_white_pin), pole_white_changed, CHANGE);
        delay(1000);
        pole_rotary_index = 0;
        pole_rotary_net_degrees.number = 0.0;
        pole_velocity.number = 0.0;
        pole_acceleration.number = 0.0;
        pole_rotary_state_time_ms = millis();
      }
    }

    else if (command == GET_STATE) {
      if (identifier == cart_rotary_encoder_identifier) {
      }
      else if (identifier == pole_rotary_encoder_identifier) {
        write_float(pole_rotary_net_degrees);
        write_float(pole_velocity);
        write_float(pole_acceleration);
        write_bool(pole_rotary_clockwise);
      }
    }

    else if (command == STOP) {
      detachInterrupt(digitalPinToInterrupt(cart_rotary_white_pin));
      detachInterrupt(digitalPinToInterrupt(pole_rotary_white_pin));
    }
  }
}
