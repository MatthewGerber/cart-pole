const size_t FLOAT_BYTES_LEN = 4;
const size_t LONG_BYTES_LEN = 4;

// structure that gives simultaneous access to floating-point numbers and their underlying bytes.
typedef union
{
  float number;
  byte bytes[FLOAT_BYTES_LEN];
} floatbytes;

// top-level command:  command id and component id
const size_t CMD_BYTES_LEN = 2;
const byte CMD_INIT = 1;
const byte CMD_GET_ROTARY_STATE = 2;
const byte CMD_SET_ROTARY_NET_TOTAL_DEGREES = 3;
const byte CMD_STOP_ROTARY = 4;
const byte CMD_SET_MOTOR_SPEED = 5;
const byte CMD_ENABLE_CART_SOFT_LIMITS = 6;
const size_t CMD_ENABLE_CART_SOFT_LIMITS_ARGS_LEN = FLOAT_BYTES_LEN * 2;
const byte CMD_DISABLE_CART_SOFT_LIMITS = 7;

const size_t CMD_INIT_ROTARY_ARGS_LEN = 18;

// cart rotary encoder
const byte CART_ROTARY_ENCODER_ID = 0;
byte cart_rotary_white_pin;
byte cart_rotary_green_pin;
volatile unsigned long cart_num_phase_changes;
volatile long cart_rotary_index;
volatile bool cart_rotary_clockwise;
unsigned long cart_num_phase_changes_value;  // non-volatile global variable used to store the volatile one
unsigned int cart_rotary_phase_changes_per_rotation;
float cart_rotary_phase_changes_per_degree;
floatbytes cart_net_degrees;
floatbytes cart_net_degrees_step_size;
floatbytes cart_velocity_deg_per_sec;
floatbytes cart_velocity_step_size;
floatbytes cart_acceleration_deg_per_sec_sq;
floatbytes cart_acceleration_step_size;
unsigned int cart_state_update_hz;
unsigned long cart_state_update_interval_ms;
unsigned long cart_rotary_state_time_ms;
long cart_left_soft_limit_rotary_index;
long cart_right_soft_limit_rotary_index;
bool cart_soft_limits_enabled;
bool cart_violates_soft_limits;
bool cart_rotary_is_inited = false;

void cart_white_changed() {
  bool cart_rotary_white_pin_value = digitalRead(cart_rotary_white_pin);
  bool cart_rotary_green_pin_value = digitalRead(cart_rotary_green_pin);
  if (cart_rotary_white_pin_value == cart_rotary_green_pin_value) {
    cart_rotary_index -= 1;
    cart_rotary_clockwise = false;
  } else {
    cart_rotary_index += 1;
    cart_rotary_clockwise = true;
  }
  cart_num_phase_changes += 1;
}

// pole rotary encoder
const byte POLE_ROTARY_ENCODER_ID = 1;
byte pole_rotary_white_pin;
byte pole_rotary_green_pin;
volatile unsigned long pole_num_phase_changes;
volatile long pole_rotary_index;
volatile bool pole_rotary_clockwise;
unsigned long pole_num_phase_changes_value;  // non-volatile global variable used to store the volatile one
unsigned int pole_rotary_phase_changes_per_rotation;
float pole_rotary_phase_changes_per_degree;
floatbytes pole_net_degrees;
floatbytes pole_net_degrees_step_size;
floatbytes pole_velocity_deg_per_sec;
floatbytes pole_velocity_step_size;
floatbytes pole_acceleration_deg_per_sec_sq;
floatbytes pole_acceleration_step_size;
unsigned int pole_state_update_hz;
unsigned long pole_state_update_interval_ms;
unsigned long pole_rotary_state_time_ms;
bool pole_rotary_is_inited = false;

void pole_white_changed() {
  bool pole_rotary_white_pin_value = digitalRead(pole_rotary_white_pin);
  bool pole_rotary_green_pin_value = digitalRead(pole_rotary_green_pin);
  if (pole_rotary_white_pin_value == pole_rotary_green_pin_value) {
    pole_rotary_index -= 1;
    pole_rotary_clockwise = false;
  } else {
    pole_rotary_index += 1;
    pole_rotary_clockwise = true;
  }
  pole_num_phase_changes += 1;
}

// motor
const byte MOTOR_ID = 2;
const size_t CMD_INIT_MOTOR_ARGS_LEN = 2;
const size_t CMD_SET_MOTOR_SPEED_ARGS_LEN = 4;
byte motor_dir_pin;
bool motor_dir_pin_value;
byte motor_pwm_pin;
int motor_current_speed;
unsigned long motor_next_set_speed_promise_time_ms;
bool motor_is_inited = false;
 
void setup() {
  Serial.begin(115200, SERIAL_8N1);
}

long bytes_to_long(byte bytes[], size_t start_idx) {
  uint32_t value = ((uint32_t)bytes[start_idx]) << 24;
  value |= ((uint32_t)bytes[start_idx + 1]) << 16;
  value |= ((uint32_t)bytes[start_idx + 2]) << 8;
  value |= ((uint32_t)bytes[start_idx + 3]);
  return (int32_t)value;
}

int bytes_to_int(byte bytes[], size_t start_idx) {
  uint16_t value = ((uint16_t)bytes[start_idx]) << 8;
  value |= ((uint16_t)bytes[start_idx + 1]);
  return (int16_t)value;
}

unsigned int bytes_to_unsigned_int(byte bytes[], size_t start_idx) {
  uint16_t value = ((uint16_t)bytes[start_idx]) << 8;
  value |= ((uint16_t)bytes[start_idx + 1]);
  return value;
}

void long_to_bytes(long value, byte bytes[]) {
  bytes[0] = (byte)(value >> 24);
  bytes[1] = (byte)(value >> 16);
  bytes[2] = (byte)(value >> 8);
  bytes[3] = (byte)value;
}

void write_long(long value) {
  byte bytes[LONG_BYTES_LEN];
  long_to_bytes(value, bytes);
  Serial.write(bytes, LONG_BYTES_LEN);
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
  if (cart_rotary_is_inited) {
    unsigned long curr_time_ms = millis();
    if (curr_time_ms <= cart_rotary_state_time_ms) {
      cart_rotary_state_time_ms = curr_time_ms;
    }
    else {
      unsigned long elapsed_ms = curr_time_ms - cart_rotary_state_time_ms;
      if (elapsed_ms >= cart_state_update_interval_ms) {

        // disable interrupts to read volatile values without corruption from the isr
        noInterrupts();
        long cart_rotary_index_value = cart_rotary_index;
        cart_num_phase_changes_value = cart_num_phase_changes;
        interrupts();

        // check soft limits. if violated, stop cart and set violation flag, which prevents setting speed until soft limits are disabled.
        if (cart_soft_limits_enabled && (cart_rotary_index_value <= cart_left_soft_limit_rotary_index || cart_rotary_index_value >= cart_right_soft_limit_rotary_index)) {
          motor_current_speed = 0;
          analogWrite(motor_pwm_pin, motor_current_speed);
          cart_violates_soft_limits = true;
        }

        // smooth net degrees
        float previous_net_degrees = cart_net_degrees.number;
        float current_net_degrees = cart_rotary_index_value / cart_rotary_phase_changes_per_degree;
        cart_net_degrees.number = (1.0 - cart_net_degrees_step_size.number) * previous_net_degrees + cart_net_degrees_step_size.number * current_net_degrees;

        float elapsed_seconds = float(elapsed_ms) / 1000.0;

        // smooth velocity        
        float previous_velocity = cart_velocity_deg_per_sec.number;
        float current_velocity = (cart_net_degrees.number - previous_net_degrees) / elapsed_seconds;
        cart_velocity_deg_per_sec.number = (1.0 - cart_velocity_step_size.number) * previous_velocity + cart_velocity_step_size.number * current_velocity;

        // smooth acceleration
        float previous_acceleration = cart_acceleration_deg_per_sec_sq.number;
        float current_acceleration = (cart_velocity_deg_per_sec.number - previous_velocity) / elapsed_seconds;
        cart_acceleration_deg_per_sec_sq.number = (1.0 - cart_acceleration_step_size.number) * previous_acceleration + cart_acceleration_step_size.number * current_acceleration;

        cart_rotary_state_time_ms = curr_time_ms;
      }
    }
  }

  // update pole rotary encoder state
  if (pole_rotary_is_inited) {
    unsigned long curr_time_ms = millis();
    if (curr_time_ms <= pole_rotary_state_time_ms) {
      pole_rotary_state_time_ms = curr_time_ms;
    }
    else {
      unsigned long elapsed_ms = curr_time_ms - pole_rotary_state_time_ms;
      if (elapsed_ms >= pole_state_update_interval_ms) {

        // disable interrupts to read volatile values without corruption from the isr
        noInterrupts();
        long pole_rotary_index_value = pole_rotary_index;
        pole_num_phase_changes_value = pole_num_phase_changes;
        interrupts();

        // smooth net degrees
        float previous_net_degrees = pole_net_degrees.number;
        float current_net_degrees = pole_rotary_index_value / pole_rotary_phase_changes_per_degree;
        pole_net_degrees.number = (1.0 - pole_net_degrees_step_size.number) * previous_net_degrees + pole_net_degrees_step_size.number * current_net_degrees;

        float elapsed_seconds = float(elapsed_ms) / 1000.0;

        // smooth velocity
        float previous_velocity = pole_velocity_deg_per_sec.number;
        float current_velocity = (pole_net_degrees.number - previous_net_degrees) / elapsed_seconds;
        pole_velocity_deg_per_sec.number = (1.0 - pole_velocity_step_size.number) * previous_velocity + pole_velocity_step_size.number * current_velocity;

        // smooth acceleration
        float previous_acceleration = pole_acceleration_deg_per_sec_sq.number;
        float current_acceleration = (pole_velocity_deg_per_sec.number - previous_velocity) / elapsed_seconds;
        pole_acceleration_deg_per_sec_sq.number = (1.0 - pole_acceleration_step_size.number) * previous_acceleration + pole_acceleration_step_size.number * current_acceleration;

        pole_rotary_state_time_ms = curr_time_ms;
      }
    }
  }

  // check for a broken promise about setting the motor speed. stop motor if promise is broken.
  if (motor_is_inited && motor_next_set_speed_promise_time_ms != 0 && millis() > motor_next_set_speed_promise_time_ms) {
    motor_current_speed = 0;
    analogWrite(motor_pwm_pin, motor_current_speed);
    motor_next_set_speed_promise_time_ms = 0;
  }

  // process a command sent over the serial connection
  if (Serial.available()) {

    byte command_bytes[CMD_BYTES_LEN];
    Serial.readBytes(command_bytes, CMD_BYTES_LEN);
    byte command = command_bytes[0];
    byte component_id = command_bytes[1];

    // initialize a component
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

        set_float_bytes(cart_net_degrees_step_size.bytes, args, 5);
        set_float_bytes(cart_velocity_step_size.bytes, args, 9);
        set_float_bytes(cart_acceleration_step_size.bytes, args, 13);

        cart_state_update_hz = args[17];
        cart_state_update_interval_ms = (unsigned long) (1000.0 / float(cart_state_update_hz));

        attachInterrupt(digitalPinToInterrupt(cart_rotary_white_pin), cart_white_changed, CHANGE);
        delay(1000);  // i forget why this delay is needed...something about letting the interrupt get configured.
        cart_num_phase_changes = 0;
        cart_num_phase_changes_value = 0;
        cart_rotary_index = 0;
        cart_net_degrees.number = 0.0;
        cart_velocity_deg_per_sec.number = 0.0;
        cart_acceleration_deg_per_sec_sq.number = 0.0;
        cart_rotary_clockwise = false;
        cart_rotary_state_time_ms = millis();
        cart_rotary_is_inited = true;

        // there's some timing issue with using delay above, such that if the sender writes data 
        // while in the delay then commands from the new data can be lost. use a synchronous 
        // return value here to prevent further writes during the delay.
        write_bool(true);
        Serial.flush();
      }
      else if (component_id == POLE_ROTARY_ENCODER_ID) {
        byte args[CMD_INIT_ROTARY_ARGS_LEN];
        Serial.readBytes(args, CMD_INIT_ROTARY_ARGS_LEN);
        pole_rotary_white_pin = args[0];
        pinMode(pole_rotary_white_pin, INPUT_PULLUP);
        pole_rotary_green_pin = args[1];
        pinMode(pole_rotary_green_pin, INPUT_PULLUP);

        // todo:  2 bytes for phase changes per rotation
        pole_rotary_phase_changes_per_rotation = 1200;
        pole_rotary_phase_changes_per_degree = float(pole_rotary_phase_changes_per_rotation) / 360.0;

        // todo:  1 byte for phase-change mode

        set_float_bytes(pole_net_degrees_step_size.bytes, args, 5);
        set_float_bytes(pole_velocity_step_size.bytes, args, 9);
        set_float_bytes(pole_acceleration_step_size.bytes, args, 13);

        pole_state_update_hz = args[17];
        pole_state_update_interval_ms = (unsigned long) (1000.0 / float(pole_state_update_hz));

        attachInterrupt(digitalPinToInterrupt(pole_rotary_white_pin), pole_white_changed, CHANGE);
        delay(1000);  // i forget why this delay is needed...something about letting the interrupt get configured.
        pole_num_phase_changes = 0;
        pole_num_phase_changes_value = 0;
        pole_rotary_index = 0;
        pole_net_degrees.number = 0.0;
        pole_velocity_deg_per_sec.number = 0.0;
        pole_acceleration_deg_per_sec_sq.number = 0.0;
        pole_rotary_clockwise = false;
        pole_rotary_state_time_ms = millis();
        pole_rotary_is_inited = true;

        // there's some timing issue with using delay above, such that if the sender writes data 
        // while in the delay then commands from the new data can be lost. use a synchronous 
        // return value here to prevent further writes during the delay.
        write_bool(true);
        Serial.flush();
      }
      else if (component_id == MOTOR_ID) {

        motor_current_speed = 0;
        motor_next_set_speed_promise_time_ms = 0;

        byte args[CMD_INIT_MOTOR_ARGS_LEN];
        Serial.readBytes(args, CMD_INIT_MOTOR_ARGS_LEN);

        motor_dir_pin = args[0];
        pinMode(motor_dir_pin, OUTPUT);
        motor_dir_pin_value = HIGH;
        digitalWrite(motor_dir_pin, motor_dir_pin_value);

        motor_pwm_pin = args[1];
        pinMode(motor_pwm_pin, OUTPUT);
        analogWrite(motor_pwm_pin, motor_current_speed);
  
        motor_is_inited = true;

      }
    }
    else if (command == CMD_GET_ROTARY_STATE) {
      if (component_id == CART_ROTARY_ENCODER_ID) {
        write_long(cart_num_phase_changes_value);
        write_float(cart_net_degrees);
        write_float(cart_velocity_deg_per_sec);
        write_float(cart_acceleration_deg_per_sec_sq);
        write_bool(cart_rotary_clockwise);
        write_long(cart_rotary_state_time_ms);
        Serial.flush();
      }
      else if (component_id == POLE_ROTARY_ENCODER_ID) {
        write_long(pole_num_phase_changes_value);
        write_float(pole_net_degrees);
        write_float(pole_velocity_deg_per_sec);
        write_float(pole_acceleration_deg_per_sec_sq);
        write_bool(pole_rotary_clockwise);
        write_long(pole_rotary_state_time_ms);
        Serial.flush();
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
        cart_net_degrees.number = cart_rotary_index / cart_rotary_phase_changes_per_degree;
        cart_velocity_deg_per_sec.number = 0.0;
        cart_acceleration_deg_per_sec_sq.number = 0.0;
        cart_rotary_state_time_ms = millis();
      }
      else if (component_id == POLE_ROTARY_ENCODER_ID) {
        pole_rotary_index = (long) net_total_degrees.number * pole_rotary_phase_changes_per_degree;
        pole_net_degrees.number = pole_rotary_index / pole_rotary_phase_changes_per_degree;
        pole_velocity_deg_per_sec.number = 0.0;
        pole_acceleration_deg_per_sec_sq.number = 0.0;
        pole_rotary_state_time_ms = millis();
      }

      interrupts();

    }
    else if (command == CMD_STOP_ROTARY) {
      if (component_id == CART_ROTARY_ENCODER_ID) {
        detachInterrupt(digitalPinToInterrupt(cart_rotary_white_pin));
        cart_rotary_is_inited = false;
      }
      else if (component_id == POLE_ROTARY_ENCODER_ID) {
        detachInterrupt(digitalPinToInterrupt(pole_rotary_white_pin));
        pole_rotary_is_inited = false;
      }
    }
    else if (command == CMD_SET_MOTOR_SPEED) {

      if (component_id == MOTOR_ID) {
                
        byte args[CMD_SET_MOTOR_SPEED_ARGS_LEN];
        Serial.readBytes(args, CMD_SET_MOTOR_SPEED_ARGS_LEN);

        // only set speed if cart doesn't violate soft limits
        if (!cart_soft_limits_enabled || !cart_violates_soft_limits) {

          int new_speed = bytes_to_int(args, 0);
          unsigned int next_set_promise_ms = bytes_to_unsigned_int(args, 2);

          // if we're changing direction, set speed to zero so that changing the direction next does not then output the 
          // current speed in the opposite direction.
          if (
            ((motor_current_speed > 0) && (new_speed <= 0)) ||
            ((motor_current_speed < 0) && (new_speed >= 0))
          ) {
            analogWrite(motor_pwm_pin, 0);
          }

          // set direction if it has changed
          bool new_motor_dir_pin_value = LOW;
          if (new_speed > 0) {
            new_motor_dir_pin_value = HIGH;
          }
          if (new_motor_dir_pin_value != motor_dir_pin_value) {
            digitalWrite(motor_dir_pin, new_motor_dir_pin_value);
            motor_dir_pin_value = new_motor_dir_pin_value;
          }

          // set the duty cycle corresponding to the new speed
          analogWrite(motor_pwm_pin, byte(255.0 * abs(new_speed) / 100.0));

          // set new promise if we have one
          if (next_set_promise_ms == 0) {
            motor_next_set_speed_promise_time_ms = 0;
          }
          else {
            motor_next_set_speed_promise_time_ms = millis() + next_set_promise_ms;
          }

          motor_current_speed = new_speed;

        }
      }
    }
    else if (command == CMD_ENABLE_CART_SOFT_LIMITS) {

      byte args[CMD_ENABLE_CART_SOFT_LIMITS_ARGS_LEN];
      Serial.readBytes(args, CMD_ENABLE_CART_SOFT_LIMITS_ARGS_LEN);

      floatbytes left_soft_limit_degrees;
      set_float_bytes(left_soft_limit_degrees.bytes, args, 0);
      cart_left_soft_limit_rotary_index = (long) left_soft_limit_degrees.number * cart_rotary_phase_changes_per_degree;

      floatbytes right_soft_limit_degrees;
      set_float_bytes(right_soft_limit_degrees.bytes, args, FLOAT_BYTES_LEN);
      cart_right_soft_limit_rotary_index = (long) right_soft_limit_degrees.number * cart_rotary_phase_changes_per_degree; 
      
      cart_soft_limits_enabled = true;

    }
    else if (command == CMD_DISABLE_CART_SOFT_LIMITS) {
      cart_soft_limits_enabled = false;
      cart_violates_soft_limits = false;
    }
  }
}
