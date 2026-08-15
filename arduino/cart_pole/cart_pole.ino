// SerialUSB writes to the arduino IDE serial monitor. _UART1_ writes to the serial tx/rx gpio pins.
// define a nicer variable to refer to serial tx/rx.
#define SerialUART _UART1_

bool DEBUG = false;

const size_t FLOAT_BYTES_LEN = 4;
const float FLOAT_SCALE = 0.001;
const size_t LONG_BYTES_LEN = 4;
const byte CART_ROTARY_ENCODER_ID = 0;
const byte POLE_ROTARY_ENCODER_ID = 1;

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
  floatbytes net_degrees;
  floatbytes net_degrees_step_size;
  floatbytes velocity_deg_per_sec;
  floatbytes velocity_step_size;
  floatbytes acceleration_deg_per_sec_sq;
  floatbytes acceleration_step_size;

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
const byte CMD_INIT = 1; const size_t CMD_INIT_ROTARY_ARGS_LEN = 15;
const byte CMD_GET_ROTARY_STATE = 2; const size_t ROTARY_STATE_RESPONSE_LEN = 21;
const byte CMD_SET_ROTARY_NET_TOTAL_DEGREES = 3;
const byte CMD_STOP_ROTARY = 4;
const byte CMD_SET_MOTOR_SPEED = 5;
const byte CMD_ENABLE_CART_SOFT_LIMITS = 6; const size_t CMD_ENABLE_CART_SOFT_LIMITS_ARGS_LEN = FLOAT_BYTES_LEN * 2;
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
rotary_encoder cart_rotary;
void cart_white_changed() {
  white_changed(&cart_rotary);
}
void cart_green_changed() {
  green_changed(&cart_rotary);
}

// pole rotary encoder and isrs
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

/**
 * Initialize a rotary encoder.
 *
 * @param rotary Pointer to a rotary encoder structure.
 * @param args Initialization arguments sent by client.
*/
void init_rotary_encoder(rotary_encoder* rotary, byte args[]) {

  rotary->white_pin = args[0];
  pinMode(rotary->white_pin, INPUT_PULLUP);
  rotary->green_pin = args[1];
  pinMode(rotary->green_pin, INPUT_PULLUP);

  set_float_bytes(rotary->net_degrees_step_size.bytes, args, 2);
  set_float_bytes(rotary->velocity_step_size.bytes, args, 6);
  set_float_bytes(rotary->acceleration_step_size.bytes, args, 10);

  rotary->state_update_hz = args[14];
  rotary->state_update_interval_ms = (unsigned long) (1000.0f / float(rotary->state_update_hz));

  rotary->white_value = digitalRead(rotary->white_pin);
  rotary->green_value = digitalRead(rotary->green_pin);
  rotary->waiting_on_white = rotary->white_value == rotary->green_value;
  rotary->waiting_on_green = !rotary->waiting_on_white;
  rotary->num_phase_changes_volatile = rotary->num_phase_changes = 0;
  rotary->index_volatile = rotary->index = 0;
  rotary->clockwise_volatile = rotary->clockwise = true;
  rotary->net_degrees.number = 0.0;
  rotary->velocity_deg_per_sec.number = 0.0;
  rotary->acceleration_deg_per_sec_sq.number = 0.0;
  rotary->state_time_ms = millis();
  rotary->soft_limits_enabled = false;
  rotary->left_soft_limit_rotary_index = 0;
  rotary->right_soft_limit_rotary_index = 0;
  rotary->violates_soft_limits = false;
  rotary->is_inited = true;

  if (DEBUG) {
    SerialUSB.println(
      "Initialized rotary encoder " + String(rotary->identifier) + "\n" + 
      "\tNet degrees step size:  " + String(rotary->net_degrees_step_size.number) + "\n" + 
      "\tVelocity step size:  " + String(rotary->velocity_step_size.number) + "\n" + 
      "\tAcceleration step size:  " + String(rotary->acceleration_step_size.number) + "\n" + 
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
      float previous_net_degrees = rotary->net_degrees.number;
      float current_net_degrees = rotary->index / rotary->phase_changes_per_degree;
      rotary->net_degrees.number = (1.0 - rotary->net_degrees_step_size.number) * previous_net_degrees + rotary->net_degrees_step_size.number * current_net_degrees;        

      // smooth velocity        
      float previous_velocity = rotary->velocity_deg_per_sec.number;
      float current_velocity = (rotary->net_degrees.number - previous_net_degrees) / elapsed_seconds;
      rotary->velocity_deg_per_sec.number = (1.0 - rotary->velocity_step_size.number) * previous_velocity + rotary->velocity_step_size.number * current_velocity;

      // smooth acceleration
      float previous_acceleration = rotary->acceleration_deg_per_sec_sq.number;
      float current_acceleration = (rotary->velocity_deg_per_sec.number - previous_velocity) / elapsed_seconds;
      rotary->acceleration_deg_per_sec_sq.number = (1.0 - rotary->acceleration_step_size.number) * previous_acceleration + rotary->acceleration_step_size.number * current_acceleration;

      rotary->state_time_ms = curr_time_ms;
    }
  }
}

void write_rotary_state(rotary_encoder* rotary) {
  byte response[ROTARY_STATE_RESPONSE_LEN];
  byte four[4];
  long_to_bytes(rotary->num_phase_changes, four);
  memcpy(response, four, 4);
  long_to_bytes((long)rotary->net_degrees.number / FLOAT_SCALE, four);
  memcpy(response + 4, four, 4);
  long_to_bytes((long)rotary->velocity_deg_per_sec.number / FLOAT_SCALE, four);
  memcpy(response + 8, four, 4);
  long_to_bytes((long)rotary->acceleration_deg_per_sec_sq.number / FLOAT_SCALE, four);
  memcpy(response + 12, four, 4);
  response[16] = rotary->clockwise;
  long_to_bytes(rotary->state_time_ms, four);
  memcpy(response + 17, four, 4);
  SerialUART.write(response, ROTARY_STATE_RESPONSE_LEN);
  SerialUART.flush();
}

void set_net_total_degrees(rotary_encoder* rotary, float net_total_degrees) {
  noInterrupts();
  rotary->index = rotary->index_volatile = (long) net_total_degrees * rotary->phase_changes_per_degree;
  rotary->net_degrees.number = rotary->index / rotary->phase_changes_per_degree;
  rotary->velocity_deg_per_sec.number = 0.0;
  rotary->acceleration_deg_per_sec_sq.number = 0.0;
  rotary->state_time_ms = millis();
  interrupts();
}

void loop() {

  update_rotary_encoder_state(&cart_rotary);
  update_rotary_encoder_state(&pole_rotary);

  // check soft limits. if violated, stop cart and set violation flag, which prevents setting speed until soft limits are disabled.
  if (cart_rotary.soft_limits_enabled && (cart_rotary.index <= cart_rotary.left_soft_limit_rotary_index || cart_rotary.index >= cart_rotary.right_soft_limit_rotary_index)) {
    motor_current_speed = 0;
    analogWrite(motor_pwm_pin, motor_current_speed);
    cart_rotary.violates_soft_limits = true;
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
      float net_total_degrees = net_total_degrees_long * FLOAT_SCALE;
      if (component_id == CART_ROTARY_ENCODER_ID) {
        set_net_total_degrees(&cart_rotary, net_total_degrees);
      }
      else if (component_id == POLE_ROTARY_ENCODER_ID) {
        set_net_total_degrees(&pole_rotary, net_total_degrees);
      }
    }
    else if (command == CMD_STOP_ROTARY) {
      if (component_id == CART_ROTARY_ENCODER_ID) {
        detachInterrupt(digitalPinToInterrupt(cart_rotary.white_pin));
        detachInterrupt(digitalPinToInterrupt(cart_rotary.green_pin));
        cart_rotary.is_inited = false;
      }
      else if (component_id == POLE_ROTARY_ENCODER_ID) {
        detachInterrupt(digitalPinToInterrupt(pole_rotary.white_pin));
        detachInterrupt(digitalPinToInterrupt(pole_rotary.green_pin));
        pole_rotary.is_inited = false;
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

      floatbytes left_soft_limit_degrees;
      set_float_bytes(left_soft_limit_degrees.bytes, args, 0);
      cart_rotary.left_soft_limit_rotary_index = (long) left_soft_limit_degrees.number * cart_rotary.phase_changes_per_degree;

      floatbytes right_soft_limit_degrees;
      set_float_bytes(right_soft_limit_degrees.bytes, args, FLOAT_BYTES_LEN);
      cart_rotary.right_soft_limit_rotary_index = (long) right_soft_limit_degrees.number * cart_rotary.phase_changes_per_degree; 
      
      cart_rotary.soft_limits_enabled = true;

    }
    else if (command == CMD_DISABLE_CART_SOFT_LIMITS) {
      cart_rotary.soft_limits_enabled = false;
      cart_rotary.violates_soft_limits = false;
    }
  }
}