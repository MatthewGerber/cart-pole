// SerialUSB writes to the arduino IDE serial monitor. _UART1_ writes to the serial tx/rx gpio pins.
// define a nicer variable to refer to serial tx/rx.
#define SerialUART _UART1_

bool DEBUG = false;

const size_t FLOAT_BYTES_LEN = 4;
const size_t LONG_BYTES_LEN = 4;

// structure that gives simultaneous access to single-precision floating-point numbers and their underlying bytes
typedef union {
  float number;
  byte bytes[FLOAT_BYTES_LEN];
} floatbytes;

// reusable structure for rotary encoder configuration and state
struct rotary_encoder {

  // configuration
  byte identifier;
  unsigned int phase_changes_per_rotation = 2400;
  float phase_changes_per_degree = 2400.0 / 360.0;
  byte white_pin;
  byte green_pin;
  float float_scale;

  // volatile values updated in interrupt service routine
  volatile bool white_value;
  volatile bool waiting_on_white;
  volatile bool green_value;
  volatile bool waiting_on_green;
  volatile unsigned long num_phase_changes_volatile;
  volatile long index_volatile;
  volatile bool clockwise_volatile;

  // state values that are updated periodically
  unsigned int state_update_hz;
  unsigned long state_update_interval_ms;
  unsigned long state_time_ms;
  unsigned long num_phase_changes;
  long index;
  bool clockwise;
  float net_degrees;
  float net_degrees_step_size;
  float velocity_deg_per_sec;
  float velocity_step_size;
  float acceleration_deg_per_sec_sq;
  float acceleration_step_size;

  // soft limits -- violation flag is set if a limit is reached
  bool soft_limits_enabled;
  long left_soft_limit_rotary_index;
  long right_soft_limit_rotary_index;
  bool violates_soft_limits;

  bool is_inited = false;
};

void long_to_bytes(long value, byte bytes[]) {
  bytes[0] = (byte)(value >> 24);
  bytes[1] = (byte)(value >> 16);
  bytes[2] = (byte)(value >> 8);
  bytes[3] = (byte)value;
}

long bytes_to_long(byte bytes[], size_t start_idx) {
  uint32_t value = ((uint32_t)bytes[start_idx]) << 24;
  value |= ((uint32_t)bytes[start_idx + 1]) << 16;
  value |= ((uint32_t)bytes[start_idx + 2]) << 8;
  value |= ((uint32_t)bytes[start_idx + 3]);
  return (int32_t)value;
}

void unsigned_long_to_bytes(unsigned long value, byte bytes[]) {
  bytes[0] = (byte)(value >> 24);
  bytes[1] = (byte)(value >> 16);
  bytes[2] = (byte)(value >> 8);
  bytes[3] = (byte)value;
}

unsigned long bytes_to_unsigned_long(byte bytes[], size_t start_idx) {
  uint32_t value = ((uint32_t)bytes[start_idx]) << 24;
  value |= ((uint32_t)bytes[start_idx + 1]) << 16;
  value |= ((uint32_t)bytes[start_idx + 2]) << 8;
  value |= ((uint32_t)bytes[start_idx + 3]);
  return value;
}

void write_long(long value) {
  byte bytes[LONG_BYTES_LEN];
  long_to_bytes(value, bytes);
  SerialUART.write(bytes, LONG_BYTES_LEN);
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

void set_float_bytes(byte dest[], byte src[], size_t src_start_idx) {
  dest[0] = src[src_start_idx];
  dest[1] = src[src_start_idx + 1];
  dest[2] = src[src_start_idx + 2];
  dest[3] = src[src_start_idx + 3];
}

void write_float(floatbytes f) {
  SerialUART.write(f.bytes, FLOAT_BYTES_LEN);
}

void write_bool(bool value) {
  SerialUART.write(value);
}

// top-level command:  command id and component id
const size_t CMD_BYTES_LEN = 2;
const byte CMD_INIT = 1; 

// motor commands
const byte MOTOR_ID = 2;
const size_t CMD_INIT_MOTOR_ARGS_LEN = 2;
const byte CMD_SET_MOTOR_SPEED = 5;
const size_t CMD_SET_MOTOR_SPEED_ARGS_LEN = 4;
byte motor_dir_pin;
bool motor_dir_pin_value;
byte motor_pwm_pin;
int motor_current_speed;
unsigned long motor_next_set_speed_promise_time_ms;
bool motor_is_inited = false;

// rotary encoder commands
const size_t CMD_INIT_ROTARY_ARGS_LEN = 19;
const byte CMD_GET_ROTARY_STATE = 2; const size_t ROTARY_STATE_RESPONSE_LEN = 21;
const byte CMD_SET_ROTARY_NET_TOTAL_DEGREES = 3;
const byte CMD_STOP_ROTARY = 4;
const byte CMD_ENABLE_CART_SOFT_LIMITS = 6; const size_t CMD_ENABLE_CART_SOFT_LIMITS_ARGS_LEN = LONG_BYTES_LEN * 2;
const byte CMD_DISABLE_CART_SOFT_LIMITS = 7;

/**
 * ISR called when the white-wire signal changes.
 *
 * @param rotary Rotary encoder.
*/
void white_changed(rotary_encoder* rotary) {
  if (rotary->waiting_on_white) {
    rotary->white_value = !rotary->white_value;
    bool new_green = digitalRead(rotary->green_pin);
    bool changed_direction = new_green != rotary->green_value;
    rotary->green_value = new_green;
    if (changed_direction) {
      rotary->clockwise_volatile = !rotary->clockwise_volatile;
      rotary->index_volatile += rotary->clockwise_volatile ? 2 : -2;
      rotary->num_phase_changes_volatile += 2;
    }
    else {
      rotary->index_volatile += rotary->clockwise_volatile ? 1 : -1;
      rotary->num_phase_changes_volatile += 1;
    }
    rotary->waiting_on_white = false;
    rotary->waiting_on_green = true;
  }
}

/**
 * ISR called when the green-wire signal changes.
 *
 * @param rotary Rotary encoder.
*/
void green_changed(rotary_encoder* rotary) {
  if (rotary->waiting_on_green) {
    rotary->green_value = !rotary->green_value;
    bool new_white = digitalRead(rotary->white_pin);
    bool changed_direction = new_white != rotary->white_value;
    rotary->white_value = new_white;
    if (changed_direction) {
      rotary->clockwise_volatile = !rotary->clockwise_volatile;
      rotary->index_volatile += rotary->clockwise_volatile ? 2 : -2;
      rotary->num_phase_changes_volatile += 2;
    }
    else {
      rotary->index_volatile += rotary->clockwise_volatile ? 1 : -1;
      rotary->num_phase_changes_volatile += 1;
    }
    rotary->waiting_on_green = false;
    rotary->waiting_on_white = true;
  }
}

// cart rotary encoder and isrs
const byte CART_ROTARY_ENCODER_ID = 0;
rotary_encoder cart_rotary;
void cart_white_changed() {
  white_changed(&cart_rotary);
}
void cart_green_changed() {
  green_changed(&cart_rotary);
}

// pole rotary encoder and isrs
const byte POLE_ROTARY_ENCODER_ID = 1;
rotary_encoder pole_rotary;
void pole_white_changed() {
  white_changed(&pole_rotary);
}
void pole_green_changed() {
  green_changed(&pole_rotary);
}

void setup() {

  cart_rotary.identifier = CART_ROTARY_ENCODER_ID;
  pole_rotary.identifier = POLE_ROTARY_ENCODER_ID;

  if (DEBUG) {
    SerialUSB.begin(9600);
  }

  SerialUART.begin(115200, SERIAL_8N1);

}

/**
 * Initialize a rotary encoder.
 *
 * @param rotary Pointer to a rotary encoder structure.
 * @param args Initialization arguments sent by client.
*/
void init_rotary_encoder(rotary_encoder* rotary, byte args[]) {

  floatbytes f;

  // extract arguments
  rotary->white_pin = args[0];
  pinMode(rotary->white_pin, INPUT_PULLUP);
  rotary->green_pin = args[1];
  pinMode(rotary->green_pin, INPUT_PULLUP);

  set_float_bytes(f.bytes, args, 2);
  rotary->net_degrees_step_size = f.number;

  set_float_bytes(f.bytes, args, 6);
  rotary->velocity_step_size = f.number;

  set_float_bytes(f.bytes, args, 10);
  rotary->acceleration_step_size = f.number;

  rotary->float_scale = float(bytes_to_unsigned_long(args, 14));
  rotary->state_update_hz = args[18];

  // initialize the rotary encoder
  rotary->state_update_interval_ms = (unsigned long) (1000.0f / float(rotary->state_update_hz));
  rotary->white_value = digitalRead(rotary->white_pin);
  rotary->green_value = digitalRead(rotary->green_pin);
  rotary->waiting_on_white = rotary->white_value == rotary->green_value;
  rotary->waiting_on_green = !rotary->waiting_on_white;
  rotary->num_phase_changes_volatile = rotary->num_phase_changes = 0;
  rotary->index_volatile = rotary->index = 0;
  rotary->clockwise_volatile = rotary->clockwise = true;
  rotary->net_degrees = 0.0;
  rotary->velocity_deg_per_sec = 0.0;
  rotary->acceleration_deg_per_sec_sq = 0.0;
  rotary->state_time_ms = millis();
  rotary->soft_limits_enabled = false;
  rotary->left_soft_limit_rotary_index = 0;
  rotary->right_soft_limit_rotary_index = 0;
  rotary->violates_soft_limits = false;
  rotary->is_inited = true;

  if (DEBUG) {
    SerialUSB.println(
      "Initialized rotary encoder " + String(rotary->identifier) + ":\n" + 
      "\tNet degrees step size:  " + String(rotary->net_degrees_step_size) + "\n" + 
      "\tVelocity step size:  " + String(rotary->velocity_step_size) + "\n" + 
      "\tAcceleration step size:  " + String(rotary->acceleration_step_size) + "\n" + 
      "\tFloat scaling:  " + String(rotary->float_scale) + "\n" + 
      "\tUpdate interval (ms):  " + String(rotary->state_update_interval_ms)
    );
  }
}

/**
 * Update a rotary encoder's state.
 *
 * @param rotary Rotary encoder.
*/
void update_rotary_encoder_state(rotary_encoder* rotary) {
  if (rotary->is_inited) {
    unsigned long curr_time_ms = millis();
    unsigned long elapsed_ms = curr_time_ms - rotary->state_time_ms;
    if (elapsed_ms >= rotary->state_update_interval_ms) {

      float elapsed_seconds = elapsed_ms / 1000.0;

      // disable interrupts to read volatile values without corruption from the isr
      noInterrupts();
      rotary->num_phase_changes = rotary->num_phase_changes_volatile;
      rotary->index = rotary->index_volatile;
      rotary->clockwise = rotary->clockwise_volatile;
      interrupts();

      // smooth net degrees
      float previous_net_degrees = rotary->net_degrees;
      float current_net_degrees = rotary->index / rotary->phase_changes_per_degree;
      rotary->net_degrees = (1.0 - rotary->net_degrees_step_size) * previous_net_degrees + rotary->net_degrees_step_size * current_net_degrees;        

      // smooth velocity        
      float previous_velocity = rotary->velocity_deg_per_sec;
      float current_velocity = (rotary->net_degrees - previous_net_degrees) / elapsed_seconds;
      rotary->velocity_deg_per_sec = (1.0 - rotary->velocity_step_size) * previous_velocity + rotary->velocity_step_size * current_velocity;

      // smooth acceleration
      float previous_acceleration = rotary->acceleration_deg_per_sec_sq;
      float current_acceleration = (rotary->velocity_deg_per_sec - previous_velocity) / elapsed_seconds;
      rotary->acceleration_deg_per_sec_sq = (1.0 - rotary->acceleration_step_size) * previous_acceleration + rotary->acceleration_step_size * current_acceleration;

      rotary->state_time_ms = curr_time_ms;
    }
  }
}

/**
 * Wrapper of memcpy that returns the next starting index to write.
 *
 * @param dest Destination array.
 * @param start Start index within destination array to write.
 * @param data Data to write.
 * @param data_len Length of data to write.
 * @return Next starting index within the destination array.
*/
size_t memcpy_wrap(byte dest[], size_t start, byte data[], size_t data_len) {
  memcpy(dest + start, data, data_len);
  return start + data_len;
}

/**
 * Write rotary state to the serial connection.
 *
 * @param rotary Rotary encoder whose state should be written.
*/
void write_rotary_state(rotary_encoder* rotary) {

  byte data[ROTARY_STATE_RESPONSE_LEN];
  size_t data_idx = 0;

  byte four_bytes[4];
  unsigned_long_to_bytes(rotary->num_phase_changes, four_bytes);
  data_idx = memcpy_wrap(data, data_idx, four_bytes, 4);

  long_to_bytes(long(rotary->net_degrees * rotary->float_scale), four_bytes);
  data_idx = memcpy_wrap(data, data_idx, four_bytes, 4);

  long_to_bytes(long(rotary->velocity_deg_per_sec * rotary->float_scale), four_bytes);
  data_idx = memcpy_wrap(data, data_idx, four_bytes, 4);

  long_to_bytes(long(rotary->acceleration_deg_per_sec_sq * rotary->float_scale), four_bytes);
  data_idx = memcpy_wrap(data, data_idx, four_bytes, 4);

  data[data_idx] = rotary->clockwise;
  data_idx += 1;

  unsigned_long_to_bytes(rotary->state_time_ms, four_bytes);
  data_idx = memcpy_wrap(data, data_idx, four_bytes, 4);

  if (data_idx == ROTARY_STATE_RESPONSE_LEN) {
    SerialUART.write(data, ROTARY_STATE_RESPONSE_LEN);
    SerialUART.flush();
  }
  else if (DEBUG) {
    SerialUSB.println("Rotary state data index/length mismatch.");
  }
}

/**
 * Set net total degrees on a rotary encoder.
 *
 * @param rotary Rotary encoder to set.
 * @param net_total_degrees Degrees to set (scaled).
*/
void set_net_total_degrees(rotary_encoder* rotary, long net_total_degrees_long) {
  float net_total_degrees = net_total_degrees_long / rotary->float_scale;
  noInterrupts();
  rotary->index = rotary->index_volatile = (long) net_total_degrees * rotary->phase_changes_per_degree;
  rotary->net_degrees = rotary->index / rotary->phase_changes_per_degree;
  rotary->velocity_deg_per_sec = 0.0;
  rotary->acceleration_deg_per_sec_sq = 0.0;
  rotary->state_time_ms = millis();
  interrupts();
  if (DEBUG) {
    SerialUSB.println(
      "Set net total degrees on rotary " + String(rotary->identifier) + ":\n" +
      "\tIndex:  " + String(rotary->index) + "\n" +
      "\tNet total degrees:  " + String(rotary->net_degrees)
    );
  }
}

/**
 * Stop a rotary encoder.
 * 
 * @param rotary Rotary encoder to stop.
*/
void stop_rotary(rotary_encoder* rotary) {
  if (DEBUG) {
    SerialUSB.println("Stopping rotary encoder " + String(rotary->identifier) + ".");
  }
  detachInterrupt(digitalPinToInterrupt(rotary->white_pin));
  detachInterrupt(digitalPinToInterrupt(rotary->green_pin));
  rotary->is_inited = false;
}

void loop() {

  update_rotary_encoder_state(&cart_rotary);
  update_rotary_encoder_state(&pole_rotary);

  // check soft limits. if violated, stop cart and set violation flag, which prevents setting speed until soft limits are disabled.
  if (cart_rotary.soft_limits_enabled && (cart_rotary.index <= cart_rotary.left_soft_limit_rotary_index || cart_rotary.index >= cart_rotary.right_soft_limit_rotary_index)) {
    cart_rotary.violates_soft_limits = true;
    if (motor_current_speed != 0) {
      motor_current_speed = 0;
      analogWrite(motor_pwm_pin, motor_current_speed);
      if (DEBUG) {
        SerialUSB.println("Cart violated soft limits. Stopped cart.");
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
  if (SerialUART.available()) {

    byte command_bytes[CMD_BYTES_LEN];
    SerialUART.readBytes(command_bytes, CMD_BYTES_LEN);
    byte command = command_bytes[0];
    byte component_id = command_bytes[1];

    if (DEBUG) {
      SerialUSB.println("Processing new command " + String(command) + " for component " + String(component_id) + ".");
    }

    // initialize a component
    if (command == CMD_INIT) {
      if (component_id == CART_ROTARY_ENCODER_ID) {
        byte args[CMD_INIT_ROTARY_ARGS_LEN];
        SerialUART.readBytes(args, CMD_INIT_ROTARY_ARGS_LEN);
        init_rotary_encoder(&cart_rotary, args);
        attachInterrupt(digitalPinToInterrupt(cart_rotary.white_pin), cart_white_changed, CHANGE);
        attachInterrupt(digitalPinToInterrupt(cart_rotary.green_pin), cart_green_changed, CHANGE);
      }
      else if (component_id == POLE_ROTARY_ENCODER_ID) {
        byte args[CMD_INIT_ROTARY_ARGS_LEN];
        SerialUART.readBytes(args, CMD_INIT_ROTARY_ARGS_LEN);
        init_rotary_encoder(&pole_rotary, args);
        attachInterrupt(digitalPinToInterrupt(pole_rotary.white_pin), pole_white_changed, CHANGE);
        attachInterrupt(digitalPinToInterrupt(pole_rotary.green_pin), pole_green_changed, CHANGE);
      }
      else if (component_id == MOTOR_ID) {
        byte args[CMD_INIT_MOTOR_ARGS_LEN];
        SerialUART.readBytes(args, CMD_INIT_MOTOR_ARGS_LEN);
        motor_current_speed = 0;
        motor_next_set_speed_promise_time_ms = 0;

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
        write_rotary_state(&cart_rotary);
      }
      else if (component_id == POLE_ROTARY_ENCODER_ID) {
        write_rotary_state(&pole_rotary);
      }
    }
    else if (command == CMD_SET_ROTARY_NET_TOTAL_DEGREES) {
      byte args[LONG_BYTES_LEN];
      SerialUART.readBytes(args, LONG_BYTES_LEN);
      long net_total_degrees_long = bytes_to_long(args, 0);
      if (component_id == CART_ROTARY_ENCODER_ID) {
        set_net_total_degrees(&cart_rotary, net_total_degrees_long);
      }
      else if (component_id == POLE_ROTARY_ENCODER_ID) {
        set_net_total_degrees(&pole_rotary, net_total_degrees_long);
      }
    }
    else if (command == CMD_STOP_ROTARY) {
      if (component_id == CART_ROTARY_ENCODER_ID) {
        stop_rotary(&cart_rotary);
      }
      else if (component_id == POLE_ROTARY_ENCODER_ID) {
        stop_rotary(&pole_rotary);
      }
    }
    else if (command == CMD_SET_MOTOR_SPEED) {

      if (component_id == MOTOR_ID) {
                
        byte args[CMD_SET_MOTOR_SPEED_ARGS_LEN];
        SerialUART.readBytes(args, CMD_SET_MOTOR_SPEED_ARGS_LEN);

        // only set speed if cart doesn't violate soft limits
        if (!cart_rotary.soft_limits_enabled || !cart_rotary.violates_soft_limits) {

          int new_speed = bytes_to_int(args, 0);
          unsigned int next_set_promise_ms = bytes_to_unsigned_int(args, 2);

          // if we're changing direction, set speed to zero so that changing the direction 
          // next does not then output the current speed in the opposite direction.
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
      SerialUART.readBytes(args, CMD_ENABLE_CART_SOFT_LIMITS_ARGS_LEN);
      float left_soft_limit_degrees = bytes_to_long(args, 0) / cart_rotary.float_scale;
      cart_rotary.left_soft_limit_rotary_index = (long) left_soft_limit_degrees * cart_rotary.phase_changes_per_degree;
      float right_soft_limit_degrees = bytes_to_long(args, 4) / cart_rotary.float_scale;
      cart_rotary.right_soft_limit_rotary_index = (long) right_soft_limit_degrees * cart_rotary.phase_changes_per_degree; 
      cart_rotary.soft_limits_enabled = true;
      if (DEBUG) {
        SerialUSB.println(
          "Enabled cart soft limits:\n"
          "\tLeft:  " + String(left_soft_limit_degrees) + " deg; index " + String(cart_rotary.left_soft_limit_rotary_index) + "\n" + 
          "\tRight:  " + String(right_soft_limit_degrees) + " deg; index " + String(cart_rotary.right_soft_limit_rotary_index)
        );
      }
    }
    else if (command == CMD_DISABLE_CART_SOFT_LIMITS) {
      cart_rotary.soft_limits_enabled = false;
      cart_rotary.violates_soft_limits = false;
      if (DEBUG) {
        SerialUSB.println("Disabled cart soft limits.");
      }
    }
  }
}