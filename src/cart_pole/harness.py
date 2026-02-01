import logging
import time

import RPi.GPIO as gpio
import serial
from matplotlib import pyplot as plt
from raspberry_py.utils import get_bytes
from serial import Serial
from smbus2 import SMBus

from raspberry_py.gpio import setup, cleanup, CkPin
from raspberry_py.gpio.communication import LockingSerial
from raspberry_py.gpio.controls import LimitSwitch
from raspberry_py.gpio.integrated_circuits import PulseWaveModulatorPCA9685PW
from raspberry_py.gpio.lights import LED
from raspberry_py.gpio.motors import DcMotor, DcMotorDriverIndirectArduino, Servo, Sg90DriverPCA9685PW
from raspberry_py.gpio.sensors import RotaryEncoder, UltrasonicRangeFinder

from cart_pole.environment import ArduinoCommand


def main():

    logging.getLogger().setLevel(logging.DEBUG)

    setup()

    locking_serial = LockingSerial(
        connection=Serial(
            port='/dev/ttyAMA0',
            baudrate=115200,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS
        ),
        throughput_step_size=0.05,
        manual_buffer=False
    )

    # basic write/read
    # x = 1.0
    # while True:
    #     with locking_serial.lock:
    #         locking_serial.connection.write(RotaryEncoder.Arduino.get_bytes(x))
    #         x = RotaryEncoder.Arduino.get_float(locking_serial.connection.read(4))
    #         print(f'{x}')
    #         time.sleep(1.0)
    #
    # arduino code
    #
    # if (Serial.available()) {
    #   byte bytes[4];
    #   Serial.readBytes(bytes, 4);
    #   floatbytes value;
    #   set_float_bytes(value.bytes, bytes, 0);
    #   value.number += 1.0;
    #   write_float(value);
    # }
    # return;

    cart_rotary_interface = RotaryEncoder.Arduino(
        phase_a_pin=2,
        phase_b_pin=4,
        phase_changes_per_rotation=2400,
        phase_change_mode=RotaryEncoder.PhaseChangeMode.TWO_SIGNAL_TWO_EDGE,
        angle_step_size=0.9,
        angular_velocity_step_size=0.5,
        angular_acceleration_step_size=0.1,
        serial=locking_serial,
        identifier=0,
        state_update_hz=round(1.5 * 45.0)
    )
    cart_rotary_encoder = RotaryEncoder(
        interface=cart_rotary_interface
    )
    cart_rotary_encoder.start()

    pole_rotary_interface = RotaryEncoder.Arduino(
        phase_a_pin=3,
        phase_b_pin=5,
        phase_changes_per_rotation=2400,
        phase_change_mode=RotaryEncoder.PhaseChangeMode.TWO_SIGNAL_TWO_EDGE,
        angle_step_size=0.9,
        angular_velocity_step_size=0.5,
        angular_acceleration_step_size=0.1,
        serial=locking_serial,
        identifier=1,
        state_update_hz=round(1.5 * 45.0)
    )
    pole_rotary_encoder = RotaryEncoder(
        interface=pole_rotary_interface
    )
    pole_rotary_encoder.start()

    motor_driver = DcMotorDriverIndirectArduino(
        identifier=2,
        serial=locking_serial,
        arduino_direction_pin=12,
        arduino_pwm_pin=9,
        next_set_speed_promise_ms=100,
        reverse=True
    )
    motor = DcMotor(
        driver=motor_driver,
        speed=0
    )
    gpio.setup(CkPin.GPIO6, gpio.OUT)

    left_limit_switch = LimitSwitch(
        input_pin=CkPin.GPIO20,
        bounce_time_ms=5
    )
    right_limit_switch = LimitSwitch(
        input_pin=CkPin.GPIO16,
        bounce_time_ms=5
    )

    led = LED(CkPin.GPIO19)

    range_finder = UltrasonicRangeFinder(
        trigger_pin=CkPin.GPIO23,
        echo_pin=CkPin.GPIO24,
        measurements_per_second=2
    )

    i2c_bus = SMBus('/dev/i2c-1')
    pwm = PulseWaveModulatorPCA9685PW(
        bus=i2c_bus,
        address=PulseWaveModulatorPCA9685PW.PCA9685PW_ADDRESS,
        frequency_hz=50
    )
    servo = Servo(
        driver=Sg90DriverPCA9685PW(
            pca9685pw=pwm,
            output_disable_pin=CkPin.GPIO25,
            servo_channel=0,
            reverse=True,
            correction_degrees=0.0
        ),
        degrees=0.0,
        min_degree=0.0,
        max_degree=180.0
    )

    def test_pole_rotary_encoder_state():
        while True:
            time.sleep(1.0 / pole_rotary_interface.state_update_hz)
            pole_rotary_encoder.update_state()
            state: RotaryEncoder.State = pole_rotary_encoder.get_state()
            print(
                f'Num phase changes:  {state.num_phase_changes}\n'
                f'Net total degrees:  {state.net_total_degrees}\n'
                f'Degrees:  {state.degrees}\n'
                f'Clockwise:  {state.clockwise}\n'
                f'Velocity:  {state.angular_velocity} deg/s\n'
                f'Acceleration:  {state.angular_acceleration} deg/s^2\n'
            )

    def test_plot_pole_rotary_encoder_state():
        print('Will begin recording state in 5 seconds...', end='')
        time.sleep(5)
        print('recording.')
        test_start = time.time()
        times = []
        net_total_degrees = []
        degrees = []
        velocities = []
        accelerations = []
        while time.time() - test_start < 10.0:
            time.sleep(1.0 / 25.0)
            pole_rotary_encoder.update_state()
            state: RotaryEncoder.State = pole_rotary_encoder.get_state()
            times.append(state.epoch_ms)
            net_total_degrees.append(state.net_total_degrees)
            degrees.append(state.degrees)
            velocities.append(state.angular_velocity)
            accelerations.append(state.angular_acceleration)

        plt.gcf().set_size_inches(20.0, 20.0)
        plt.plot(times, net_total_degrees, label='total degrees')
        plt.plot(times, degrees, label='degrees')
        plt.plot(times, velocities, label='velocity')
        plt.plot(times, accelerations, label='acceleration')
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()

    def test_pole_rotary_encoder_wait_for_stationarity():
        while True:
            time.sleep(1.0 / pole_rotary_interface.state_update_hz)
            pole_rotary_encoder.update_state()
            state: RotaryEncoder.State = pole_rotary_encoder.get_state()
            if state.angular_velocity > 0.0:
                print(f'Angular velocity is {state.angular_velocity}. Waiting for stationarity...')
                pole_rotary_encoder.wait_for_stationarity()
            else:
                print('Rotary encoder is stationary.')
                time.sleep(1.0)

    def test_led():
        while True:
            if led.is_on():
                led.turn_off()
            else:
                led.turn_on()
            time.sleep(1.0)

    def test_motor():

        print('Turning off failsafe.')
        gpio.output(CkPin.GPIO6, gpio.LOW)

        motor.start()

        print('Accelerating from 0 to 100...', end='')
        for speed in range(0, 100):
            motor.set_speed(speed)
            time.sleep(0.05)
            print('.', end='')

        print('at maximum speed for 2 seconds.')
        time.sleep(2.0)

        print('Decelerating from 100 to 0 then -100 with a freeze in the middle....')
        motor_driver.send_promise = True
        for speed in range(100, -100, -1):
            motor.set_speed(speed)
            if speed == 50:
                print(f'Simulating freeze...', end='')
                time.sleep(2.0)
                print(f'resuming.')
            else:
                time.sleep(0.01)

        print(f'Decelerating from -100 to 0.')
        for speed in range(-100, 0):
            motor.set_speed(speed)
            time.sleep(0.01)

        motor.set_speed(0)
        motor.stop()

        print(
            f'write/s:  {locking_serial.bytes_written_per_second}; read/s:  {locking_serial.bytes_read_per_second}'
        )

    def test_limit_switches():
        left_limit_switch.events.clear()
        left_limit_switch.event(lambda s: print('left pressed') if s.pressed else print('left released'))
        right_limit_switch.events.clear()
        right_limit_switch.event(lambda s: print('right pressed') if s.pressed else print('right released'))
        print('Press left or right limit switches...')
        time.sleep(30.0)

    def test_pole_set_net_total_degrees():
        while True:
            print('Will set net total degrees to 0.0 in 5 seconds.')
            time.sleep(5.0)
            pole_rotary_encoder.set_net_total_degrees(0.0)
            print('Set to 0.0.')
            for _ in range(20):
                pole_rotary_encoder.update_state()
                state: RotaryEncoder.State = pole_rotary_encoder.get_state()
                print(
                    f'Num phase changes:  {state.num_phase_changes}\n'
                    f'Net total degrees:  {state.net_total_degrees}\n'
                    f'Degrees:  {state.degrees}\n'
                    f'Clockwise:  {state.clockwise}\n'
                    f'Velocity:  {state.angular_velocity} deg/s\n'
                    f'Acceleration:  {state.angular_acceleration} deg/s^2\n'
                )
                time.sleep(1.0)

    def test_range_finder():
        try:
            range_finder.event(lambda s: print(str(s)))
            range_finder.start_measuring_distance()
            time.sleep(30.0)
        except KeyboardInterrupt:
            range_finder.stop_measuring_distance()

    def test_motor_failsafe():
        try:
            gpio.output(CkPin.GPIO6, gpio.LOW)
            motor.start()
            motor.set_speed(50)
            left_limit_switch.events.clear()
            left_limit_switch.event(lambda s: (
                gpio.output(CkPin.GPIO6, gpio.HIGH) if s.pressed
                else gpio.output(CkPin.GPIO6, gpio.LOW)
            ))
            print('Press left limit switch...')
            time.sleep(30.0)
        except KeyboardInterrupt:
            motor.stop()

    def test_servo():
        release_degrees = 20.0
        braking_degrees = 12.0
        servo.disable()
        servo.start()
        servo.set_degrees(release_degrees)
        servo.enable()
        time.sleep(1.0)
        for i in range(10):
            if i == 5:
                print('Disabling servo.')
                servo.disable()
            servo.set_degrees(braking_degrees)
            time.sleep(0.5)
            servo.set_degrees(release_degrees)
            time.sleep(0.5)
            if i == 5:
                print('Enabling servo.')
                servo.enable()
        servo.stop()

    def test_arduino_soft_limits():

        def enable_cart_soft_limits():
            print('Setting degrees to 0 and limiting...', end='')
            cart_rotary_encoder.set_net_total_degrees(0.0)
            locking_serial.write_then_read(
                ArduinoCommand.ENABLE_CART_SOFT_LIMITS.to_bytes(1) +
                (0).to_bytes(1) +  # ignored
                get_bytes(-360.0) +
                get_bytes(360.0),
                True,
                0,
                False
            )
            print('done.')

        def disable_cart_soft_limits():
            print('Disabling soft limits...', end='')
            locking_serial.write_then_read(
                ArduinoCommand.DISABLE_CART_SOFT_LIMITS.to_bytes(1) +
                (0).to_bytes(1),  # ignored
                True,
                0,
                False
            )
            cart_rotary_encoder.set_net_total_degrees(0.0)
            print('done.')

        time.sleep(3.0)

        gpio.output(CkPin.GPIO6, gpio.LOW)
        motor.start()
        enable_cart_soft_limits()

        print('Setting motor speed to 20 for 10 seconds...', end='')
        motor.set_speed(20)
        time.sleep(10.0)
        print('done. It should have turned once.')

        disable_cart_soft_limits()
        enable_cart_soft_limits()

        print('Setting motor speed to -20 for 10 seconds...', end='')
        motor.set_speed(-20)
        time.sleep(10.0)
        print('done. It should have turned once.')

        disable_cart_soft_limits()
        motor.stop()

    try:
        print('Running test...')

        # test_led()
        # test_range_finder()
        # test_limit_switches()

        # motor tests
        # test_motor()
        # test_motor_failsafe()

        # rotary encoder tests
        # test_pole_rotary_encoder_state()
        # test_pole_set_net_total_degrees()
        test_plot_pole_rotary_encoder_state()
        # test_pole_rotary_encoder_wait_for_stationarity()
        # test_servo()
        # test_arduino_soft_limits()

    except KeyboardInterrupt:
        pass

    cart_rotary_encoder.cleanup()
    pole_rotary_encoder.cleanup()
    locking_serial.connection.close()
    cleanup()


if __name__ == '__main__':
    main()
