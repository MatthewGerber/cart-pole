size_t COMMAND_BYTES_LEN = 2;
size_t START_BYTES_LEN = 14;
const size_t FLOAT_BYTES_LEN = 4;

typedef union
{
  float number;
  byte bytes[FLOAT_BYTES_LEN];
} floatbytes;

byte cart_rotary_encoder_identifier = 0;
byte cart_rotary_white_pin;
byte cart_rotary_green_pin;
volatile unsigned long cart_num_phase_changes;
volatile long cart_rotary_index;
volatile bool cart_rotary_clockwise;
unsigned int cart_rotary_phase_changes_per_rotation;
float cart_rotary_phase_changes_per_degree;
floatbytes cart_rotary_net_degrees;
floatbytes cart_velocity;
floatbytes cart_velocity_step_size;
floatbytes cart_acceleration;
floatbytes cart_acceleration_step_size;
unsigned int cart_state_update_hz;
float cart_state_update_interval_ms;
unsigned long cart_rotary_state_time_ms;

byte pole_rotary_encoder_identifier = 1;
byte pole_rotary_white_pin;
byte pole_rotary_green_pin;
volatile unsigned long pole_num_phase_changes;
volatile long pole_rotary_index;
volatile bool pole_rotary_clockwise;
unsigned int pole_rotary_phase_changes_per_rotation;
float pole_rotary_phase_changes_per_degree;
floatbytes pole_rotary_net_degrees;
floatbytes pole_velocity;
floatbytes pole_velocity_step_size;
floatbytes pole_acceleration;
floatbytes pole_acceleration_step_size;
unsigned int pole_state_update_hz;
float pole_state_update_interval_ms;
unsigned long pole_rotary_state_time_ms;

byte START = 1;
byte GET_STATE = 2;
byte SET_NET_TOTAL_DEGREES = 3;
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
  cart_num_phase_changes += 1;
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
  pole_num_phase_changes += 1;
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
  byte bytes[FLOAT_BYTES_LEN];
  long_to_bytes(value, bytes);
  Serial.write(bytes, FLOAT_BYTES_LEN);
}

void write_float(floatbytes f) {
  Serial.write(f.bytes, FLOAT_BYTES_LEN);
}

void write_bool(bool value) {
  Serial.write(value);
}

void set_float_bytes(byte dest[], byte src[], size_t src_start_idx) {
  dest[0] = src[src_start_idx];
  dest[1] = src[src_start_idx + 1];
  dest[2] = src[src_start_idx + 2];
  dest[3] = src[src_start_idx + 3];
}

void loop() {

  unsigned long curr_time_ms = millis();

  unsigned long cart_rotary_state_elapsed_ms = curr_time_ms - cart_rotary_state_time_ms;
  if (cart_rotary_state_elapsed_ms >= cart_state_update_interval_ms) {
    float net_total_degrees = cart_rotary_index / cart_rotary_phase_changes_per_degree;
    float elapsed_seconds = cart_rotary_state_elapsed_ms / 1000.0;
    float current_cart_velocity = (net_total_degrees - cart_rotary_net_degrees.number) / elapsed_seconds;
    cart_rotary_net_degrees.number = net_total_degrees;
    float previous_cart_velocity = cart_velocity.number;
    cart_velocity.number = (1.0 - cart_velocity_step_size.number) * previous_cart_velocity + cart_velocity_step_size.number * current_cart_velocity;
    float current_cart_acceleration = (cart_velocity.number - previous_cart_velocity) / elapsed_seconds;
    cart_acceleration.number = (1.0 - cart_acceleration_step_size.number) * cart_acceleration.number + cart_acceleration_step_size.number * current_cart_acceleration;
    cart_rotary_state_time_ms = curr_time_ms;
  }

  unsigned long pole_rotary_state_elapsed_ms = curr_time_ms - pole_rotary_state_time_ms;
  if (pole_rotary_state_elapsed_ms >= pole_state_update_interval_ms) {
    float net_total_degrees = pole_rotary_index / pole_rotary_phase_changes_per_degree;
    float elapsed_seconds = pole_rotary_state_elapsed_ms / 1000.0;
    float current_pole_velocity = (net_total_degrees - pole_rotary_net_degrees.number) / elapsed_seconds;
    pole_rotary_net_degrees.number = net_total_degrees;
    float previous_pole_velocity = pole_velocity.number;
    pole_velocity.number = (1.0 - pole_velocity_step_size.number) * previous_pole_velocity + pole_velocity_step_size.number * current_pole_velocity;
    float current_pole_acceleration = (pole_velocity.number - previous_pole_velocity) / elapsed_seconds;
    pole_acceleration.number = (1.0 - pole_acceleration_step_size.number) * pole_acceleration.number + pole_acceleration_step_size.number * current_pole_acceleration;
    pole_rotary_state_time_ms = curr_time_ms;
  }

  if (Serial.available()) {

    byte command_bytes[COMMAND_BYTES_LEN];
    Serial.readBytes(command_bytes, COMMAND_BYTES_LEN);
    byte command = command_bytes[0];
    byte identifier = command_bytes[1];

    if (command == START) {

      byte subcommand_bytes[START_BYTES_LEN];
      Serial.readBytes(subcommand_bytes, START_BYTES_LEN);

      if (identifier == cart_rotary_encoder_identifier) {
        cart_rotary_white_pin = subcommand_bytes[0];
        pinMode(cart_rotary_white_pin, INPUT_PULLUP);
        digitalWrite(cart_rotary_white_pin, HIGH);
        cart_rotary_green_pin = subcommand_bytes[1];
        pinMode(cart_rotary_green_pin, INPUT_PULLUP);
        digitalWrite(cart_rotary_green_pin, HIGH);

        // todo:  2 bytes for phase changes per rotation
        cart_rotary_phase_changes_per_rotation = 1200;
        cart_rotary_phase_changes_per_degree = cart_rotary_phase_changes_per_rotation / 360.0;

        // todo:  1 byte for phase-change mode

        set_float_bytes(cart_velocity_step_size.bytes, subcommand_bytes, 5);
        set_float_bytes(cart_acceleration_step_size.bytes, subcommand_bytes, 9);

        cart_state_update_hz = subcommand_bytes[13];
        cart_state_update_interval_ms = 1000.0 / (float)cart_state_update_hz;

        attachInterrupt(digitalPinToInterrupt(cart_rotary_white_pin), cart_white_changed, CHANGE);
        delay(1000);
        cart_num_phase_changes = 0;
        cart_rotary_index = 0;
        cart_rotary_net_degrees.number = 0.0;
        cart_velocity.number = 0.0;
        cart_acceleration.number = 0.0;
        cart_rotary_clockwise = false;
        cart_rotary_state_time_ms = millis();
      }
      else if (identifier == pole_rotary_encoder_identifier) {
        pole_rotary_white_pin = subcommand_bytes[0];
        pinMode(pole_rotary_white_pin, INPUT_PULLUP);
        digitalWrite(pole_rotary_white_pin, HIGH);
        pole_rotary_green_pin = subcommand_bytes[1];
        pinMode(pole_rotary_green_pin, INPUT_PULLUP);
        digitalWrite(pole_rotary_green_pin, HIGH);

        // todo:  2 bytes for phase changes per rotation
        pole_rotary_phase_changes_per_rotation = 1200;
        pole_rotary_phase_changes_per_degree = pole_rotary_phase_changes_per_rotation / 360.0;

        // todo:  1 byte for phase-change mode

        set_float_bytes(pole_velocity_step_size.bytes, subcommand_bytes, 5);
        set_float_bytes(pole_acceleration_step_size.bytes, subcommand_bytes, 9);

        pole_state_update_hz = subcommand_bytes[13];
        pole_state_update_interval_ms = 1000.0 / (float)pole_state_update_hz;

        attachInterrupt(digitalPinToInterrupt(pole_rotary_white_pin), pole_white_changed, CHANGE);
        delay(1000);
        pole_num_phase_changes = 0;
        pole_rotary_index = 0;
        pole_rotary_net_degrees.number = 0.0;
        pole_velocity.number = 0.0;
        pole_acceleration.number = 0.0;
        pole_rotary_clockwise = false;
        pole_rotary_state_time_ms = millis();
      }
    }

    else if (command == GET_STATE) {
      if (identifier == cart_rotary_encoder_identifier) {
        write_long(cart_num_phase_changes);
        write_float(cart_rotary_net_degrees);
        write_float(cart_velocity);
        write_float(cart_acceleration);
        write_bool(cart_rotary_clockwise);
      }
      else if (identifier == pole_rotary_encoder_identifier) {
        write_long(pole_num_phase_changes);
        write_float(pole_rotary_net_degrees);
        write_float(pole_velocity);
        write_float(pole_acceleration);
        write_bool(pole_rotary_clockwise);
      }
    }

    else if (command == SET_NET_TOTAL_DEGREES) {

      byte subcommand_bytes[FLOAT_BYTES_LEN];
      Serial.readBytes(subcommand_bytes, FLOAT_BYTES_LEN);
      floatbytes net_total_degrees;
      set_float_bytes(net_total_degrees.bytes, subcommand_bytes, 0);

      if (identifier == cart_rotary_encoder_identifier) {
        cart_rotary_net_degrees.number = net_total_degrees.number;
        cart_rotary_index = (long) cart_rotary_net_degrees.number * cart_rotary_phase_changes_per_degree;
      }
      else if (identifier == pole_rotary_encoder_identifier) {
        pole_rotary_net_degrees.number = net_total_degrees.number;
        pole_rotary_index = (long) pole_rotary_net_degrees.number * pole_rotary_phase_changes_per_degree;
      }
    }

    else if (command == STOP) {
      detachInterrupt(digitalPinToInterrupt(cart_rotary_white_pin));
      detachInterrupt(digitalPinToInterrupt(pole_rotary_white_pin));
    }
  }
}
