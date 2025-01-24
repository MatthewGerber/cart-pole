const size_t FLOAT_BYTES_LEN = 4;

typedef union
{
  float number;
  byte bytes[FLOAT_BYTES_LEN];
} floatbytes;

// cart rotary encoder
const byte CART_ROTARY_ENCODER_ID = 0;
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
unsigned long cart_state_update_interval_ms;
unsigned long cart_rotary_state_time_ms;

// pole rotary encoder
const byte POLE_ROTARY_ENCODER_ID = 1;
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
unsigned long pole_state_update_interval_ms;
unsigned long pole_rotary_state_time_ms;

// motor
const byte MOTOR_ID = 2;
byte motor_dir_pin;
byte motor_pwm_pin;
unsigned long motor_next_set_speed_promise_time_ms;

// top-level command:  command id and component id
const size_t CMD_BYTES_LEN = 2;

// init components
const byte CMD_INIT = 1;
const size_t CMD_INIT_ROTARY_ARGS_LEN = 14;
const size_t CMD_INIT_MOTOR_ARGS_LEN = 2;

// get/set rotary state
const byte CMD_GET_ROTARY_STATE = 2;
const byte CMD_SET_ROTARY_NET_TOTAL_DEGREES = 3;

// stop rotary updates
const byte CMD_STOP_ROTARY = 4;

// set motor speed
const byte CMD_SET_MOTOR_SPEED = 5;
const size_t CMD_SET_MOTOR_SPEED_ARGS_LEN = 3;
 
void setup() {

  Serial.begin(115200, SERIAL_8N1);

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

unsigned int bytes_to_unsigned_int(byte bytes[]) {
  unsigned int value = 0;
  value += ((unsigned int)bytes[0]) << 8;
  value += ((unsigned int)bytes[1]);
  return value;
}

int bytes_to_int(byte bytes[]) {
  int value = 0;
  value += ((int)bytes[0]) << 8;
  value += ((int)bytes[1]);
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

  // update cart rotary encoder state
  unsigned long curr_time_ms = millis();
  if (curr_time_ms <= cart_rotary_state_time_ms) {
    cart_rotary_state_time_ms = curr_time_ms;
  }
  else {
    unsigned long elapsed_ms = curr_time_ms - cart_rotary_state_time_ms;
    if (elapsed_ms >= cart_state_update_interval_ms) {
      noInterrupts();
      long cart_rotary_index_value = cart_rotary_index;
      interrupts();
      float previous_net_degrees = cart_rotary_net_degrees.number;
      cart_rotary_net_degrees.number = cart_rotary_index_value / cart_rotary_phase_changes_per_degree;
      float elapsed_seconds = float(elapsed_ms) / 1000.0;
      float previous_velocity = cart_velocity.number;
      float current_velocity = (cart_rotary_net_degrees.number - previous_net_degrees) / elapsed_seconds;
      cart_velocity.number = (1.0 - cart_velocity_step_size.number) * previous_velocity + cart_velocity_step_size.number * current_velocity;
      float previous_acceleration = cart_acceleration.number;
      float current_acceleration = (cart_velocity.number - previous_velocity) / elapsed_seconds;
      cart_acceleration.number = (1.0 - cart_acceleration_step_size.number) * previous_acceleration + cart_acceleration_step_size.number * current_acceleration;
      cart_rotary_state_time_ms = curr_time_ms;
    }
  }

  // update pole rotary encoder state
  curr_time_ms = millis();
  if (curr_time_ms <= pole_rotary_state_time_ms) {
    pole_rotary_state_time_ms = curr_time_ms;
  }
  else {
    unsigned long elapsed_ms = curr_time_ms - pole_rotary_state_time_ms;
    if (elapsed_ms >= pole_state_update_interval_ms) {
      noInterrupts();
      long pole_rotary_index_value = pole_rotary_index;
      interrupts();
      float previous_net_degrees = pole_rotary_net_degrees.number;
      pole_rotary_net_degrees.number = pole_rotary_index_value / pole_rotary_phase_changes_per_degree;
      float elapsed_seconds = float(elapsed_ms) / 1000.0;
      float previous_velocity = pole_velocity.number;
      float current_velocity = (pole_rotary_net_degrees.number - previous_net_degrees) / elapsed_seconds;
      pole_velocity.number = (1.0 - pole_velocity_step_size.number) * previous_velocity + pole_velocity_step_size.number * current_velocity;
      float previous_acceleration = pole_acceleration.number;
      float current_acceleration = (pole_velocity.number - previous_velocity) / elapsed_seconds;
      pole_acceleration.number = (1.0 - pole_acceleration_step_size.number) * previous_acceleration + pole_acceleration_step_size.number * current_acceleration;
      pole_rotary_state_time_ms = curr_time_ms;
    }
  }

  // check for a broken promise about setting the motor speed. stop motor if promise is broken.
  if (motor_next_set_speed_promise_time_ms != 0 && millis() > motor_next_set_speed_promise_time_ms) {
    analogWrite(motor_pwm_pin, 0);
    motor_next_set_speed_promise_time_ms = 0;
  }

  // process a command sent over the serial connection
  if (Serial.available()) {

    byte command_bytes[CMD_BYTES_LEN];
    Serial.readBytes(command_bytes, CMD_BYTES_LEN);
    byte command = command_bytes[0];
    byte component_id = command_bytes[1];

    // initialize the component
    if (command == CMD_INIT) {

      if (component_id == CART_ROTARY_ENCODER_ID) {
        byte args[CMD_INIT_ROTARY_ARGS_LEN];
        Serial.readBytes(args, CMD_INIT_ROTARY_ARGS_LEN);
        cart_rotary_white_pin = args[0];
        pinMode(cart_rotary_white_pin, INPUT_PULLUP);
        digitalWrite(cart_rotary_white_pin, HIGH);
        cart_rotary_green_pin = args[1];
        pinMode(cart_rotary_green_pin, INPUT_PULLUP);
        digitalWrite(cart_rotary_green_pin, HIGH);

        // todo:  2 bytes for phase changes per rotation
        cart_rotary_phase_changes_per_rotation = 1200;
        cart_rotary_phase_changes_per_degree = float(cart_rotary_phase_changes_per_rotation) / 360.0;

        // todo:  1 byte for phase-change mode

        set_float_bytes(cart_velocity_step_size.bytes, args, 5);
        set_float_bytes(cart_acceleration_step_size.bytes, args, 9);

        cart_state_update_hz = args[13];
        cart_state_update_interval_ms = (unsigned long) (1000.0 / float(cart_state_update_hz));

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
      else if (component_id == POLE_ROTARY_ENCODER_ID) {
        byte args[CMD_INIT_ROTARY_ARGS_LEN];
        Serial.readBytes(args, CMD_INIT_ROTARY_ARGS_LEN);
        pole_rotary_white_pin = args[0];
        pinMode(pole_rotary_white_pin, INPUT_PULLUP);
        digitalWrite(pole_rotary_white_pin, HIGH);
        pole_rotary_green_pin = args[1];
        pinMode(pole_rotary_green_pin, INPUT_PULLUP);
        digitalWrite(pole_rotary_green_pin, HIGH);

        // todo:  2 bytes for phase changes per rotation
        pole_rotary_phase_changes_per_rotation = 1200;
        pole_rotary_phase_changes_per_degree = float(pole_rotary_phase_changes_per_rotation) / 360.0;

        // todo:  1 byte for phase-change mode

        set_float_bytes(pole_velocity_step_size.bytes, args, 5);
        set_float_bytes(pole_acceleration_step_size.bytes, args, 9);

        pole_state_update_hz = args[13];
        pole_state_update_interval_ms = (unsigned long) (1000.0 / float(pole_state_update_hz));

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
      else if (component_id == MOTOR_ID) {
        byte args[CMD_INIT_MOTOR_ARGS_LEN];
        Serial.readBytes(args, CMD_INIT_MOTOR_ARGS_LEN);

        motor_dir_pin = args[0];
        pinMode(motor_dir_pin, OUTPUT);
        digitalWrite(motor_dir_pin, LOW);

        motor_pwm_pin = args[1];
        pinMode(motor_pwm_pin, OUTPUT);
        analogWrite(motor_pwm_pin, 0);

        motor_next_set_speed_promise_time_ms = 0;
      }
    }

    else if (command == CMD_GET_ROTARY_STATE) {
      if (component_id == CART_ROTARY_ENCODER_ID) {
        write_long(cart_num_phase_changes);
        write_float(cart_rotary_net_degrees);
        write_float(cart_velocity);
        write_float(cart_acceleration);
        write_bool(cart_rotary_clockwise);
        write_long(cart_rotary_state_time_ms);
      }
      else if (component_id == POLE_ROTARY_ENCODER_ID) {
        write_long(pole_num_phase_changes);
        write_float(pole_rotary_net_degrees);
        write_float(pole_velocity);
        write_float(pole_acceleration);
        write_bool(pole_rotary_clockwise);
        write_long(pole_rotary_state_time_ms);
      }
    }

    else if (command == CMD_SET_ROTARY_NET_TOTAL_DEGREES) {

      byte args[FLOAT_BYTES_LEN];
      Serial.readBytes(args, FLOAT_BYTES_LEN);

      noInterrupts();

      floatbytes net_total_degrees;
      set_float_bytes(net_total_degrees.bytes, args, 0);

      if (component_id == CART_ROTARY_ENCODER_ID) {
        cart_rotary_index = (long) net_total_degrees.number * cart_rotary_phase_changes_per_degree;
        cart_rotary_net_degrees.number = cart_rotary_index / cart_rotary_phase_changes_per_degree;
        cart_rotary_state_time_ms = millis();
      }
      else if (component_id == POLE_ROTARY_ENCODER_ID) {
        pole_rotary_index = (long) net_total_degrees.number * pole_rotary_phase_changes_per_degree;
        pole_rotary_net_degrees.number = pole_rotary_index / pole_rotary_phase_changes_per_degree;;
        pole_rotary_state_time_ms = millis();
      }

      interrupts();

    }

    else if (command == CMD_STOP_ROTARY) {
      detachInterrupt(digitalPinToInterrupt(cart_rotary_white_pin));
      detachInterrupt(digitalPinToInterrupt(pole_rotary_white_pin));
    }

    else if (command == CMD_SET_MOTOR_SPEED) {

      byte args[CMD_SET_MOTOR_SPEED_ARGS_LEN];
      Serial.readBytes(args, CMD_SET_MOTOR_SPEED_ARGS_LEN);

      byte duty_cycle = args[0];
      analogWrite(motor_pwm_pin, duty_cycle);

      byte next_set_promise_bytes[2];
      next_set_promise_bytes[0] = args[1];
      next_set_promise_bytes[1] = args[2];
      unsigned int next_set_promise_ms = bytes_to_unsigned_int(next_set_promise_bytes);
      if (next_set_promise_ms == 0) {
        motor_next_set_speed_promise_time_ms = 0;
      }
      else {
        motor_next_set_speed_promise_time_ms = millis() + next_set_promise_ms;
      }
    }
  }
}
