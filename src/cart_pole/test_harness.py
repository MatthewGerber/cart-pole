import time

import RPi.GPIO as gpio
import serial
from matplotlib import pyplot as plt
from serial import Serial
from smbus2 import SMBus

from raspberry_py.gpio import setup, cleanup, CkPin
from raspberry_py.gpio.communication import LockingSerial
from raspberry_py.gpio.controls import LimitSwitch
from raspberry_py.gpio.integrated_circuits import PulseWaveModulatorPCA9685PW
from raspberry_py.gpio.lights import LED
from raspberry_py.gpio.motors import DcMotor, DcMotorDriverIndirectPCA9685PW
from raspberry_py.gpio.sensors import RotaryEncoder, UltrasonicRangeFinder


def main():

    setup()

    locking_serial = LockingSerial(
        connection=Serial(
            port='/dev/ttyAMA0',
            baudrate=115200,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS
        )
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

    arduino_interface = RotaryEncoder.Arduino(
        phase_a_pin=3,
        phase_b_pin=5,
        phase_changes_per_rotation=1200,
        phase_change_mode=RotaryEncoder.PhaseChangeMode.ONE_SIGNAL_TWO_EDGE,
        angular_velocity_step_size=0.5,
        angular_acceleration_step_size=0.2,
        serial=locking_serial,
        identifier=0,
        state_update_hz=50
    )
    rotary_encoder = RotaryEncoder(
        interface=arduino_interface
    )
    rotary_encoder.start()
    pca9685pw = PulseWaveModulatorPCA9685PW(
        bus=SMBus('/dev/i2c-1'),
        address=PulseWaveModulatorPCA9685PW.PCA9685PW_ADDRESS,
        frequency_hz=400
    )
    motor = DcMotor(
        driver=DcMotorDriverIndirectPCA9685PW(
            pca9685pw=pca9685pw,
            pwm_channel=0,
            direction_pin=CkPin.GPIO21
        ),
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

    def test_rotary_encoder_state():
        while True:
            time.sleep(1.0 / arduino_interface.state_update_hz)
            rotary_encoder.update_state()
            state: RotaryEncoder.State = rotary_encoder.get_state()
            print(
                f'Num phase changes:  {state.num_phase_changes}\n'
                f'Net total degrees:  {state.net_total_degrees}\n'
                f'Degrees:  {state.degrees}\n'
                f'Clockwise:  {state.clockwise}\n'
                f'Velocity:  {state.angular_velocity} deg/s\n'
                f'Acceleration:  {state.angular_acceleration} deg/s^2\n'
            )

    def test_plot_rotary_encoder_state():
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
            rotary_encoder.update_state()
            state: RotaryEncoder.State = rotary_encoder.get_state()
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

    def test_rotary_encoder_wait_for_stationarity():
        while True:
            time.sleep(1.0 / arduino_interface.state_update_hz)
            rotary_encoder.update_state()
            state: RotaryEncoder.State = rotary_encoder.get_state()
            if state.angular_velocity > 0.0:
                print(f'Angular velocity is {state.angular_velocity}. Waiting for stationarity...')
                rotary_encoder.wait_for_stationarity()
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
        gpio.output(CkPin.GPIO6, gpio.LOW)
        motor.start()
        motor.set_speed(50)
        time.sleep(1)
        motor.set_speed(-50)
        time.sleep(1)
        motor.set_speed(0)
        motor.stop()

    def test_limit_switches():
        left_limit_switch.events.clear()
        left_limit_switch.event(lambda s: print('left pressed') if s.pressed else print('left released'))
        right_limit_switch.events.clear()
        right_limit_switch.event(lambda s: print('right pressed') if s.pressed else print('right released'))
        print('Press left or right limit switches...')
        time.sleep(30.0)

    def test_set_net_total_degrees():
        while True:
            print('Will set net total degrees to 0.0 in 5 seconds.')
            time.sleep(5.0)
            rotary_encoder.set_net_total_degrees(0.0)
            print('Set to 0.0.')
            for _ in range(20):
                rotary_encoder.update_state()
                state: RotaryEncoder.State = rotary_encoder.get_state()
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

    try:
        print('Running test...')
        # test_set_net_total_degrees()
        # test_range_finder()
        # test_motor_failsafe()
        # test_limit_switches()
        # test_motor()
        # test_led()
        # test_rotary_encoder_state()
        test_plot_rotary_encoder_state()
        # test_rotary_encoder_wait_for_stationarity()

    except KeyboardInterrupt:
        pass

    rotary_encoder.cleanup()
    locking_serial.connection.close()
    cleanup()


if __name__ == '__main__':
    main()
