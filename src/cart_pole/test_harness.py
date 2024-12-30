import time

import serial
from serial import Serial
from smbus2 import SMBus
import RPi.GPIO as gpio
from raspberry_py.gpio import setup, cleanup, CkPin
from raspberry_py.gpio.integrated_circuits import PulseWaveModulatorPCA9685PW
from raspberry_py.gpio.lights import LED
from raspberry_py.gpio.motors import DcMotor, DcMotorDriverIndirectPCA9685PW
from raspberry_py.gpio.sensors import RotaryEncoder


def main():

    setup()

    locking_serial = RotaryEncoder.Arduino.LockingSerial(
        connection=Serial(
            port='/dev/ttyAMA0',
            baudrate=9600,
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
        angular_velocity_step_size=1.0,
        angular_acceleration_step_size=1.0,
        serial=locking_serial,
        identifier=1,
        state_update_hz=10
    )
    rotary_encoder = RotaryEncoder(
        interface=arduino_interface
    )
    rotary_encoder.start()

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

    def test_wait_for_stationarity():
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
        led = LED(CkPin.GPIO19)
        while True:
            time.sleep(1.0)
            if led.is_on():
                led.turn_off()
            else:
                led.turn_on()

    def test_pi_pwm_motor():

        # disable failsafe
        gpio.setup(CkPin.GPIO6, gpio.OUT)
        gpio.output(CkPin.GPIO6, gpio.LOW)

        # set up pwm chip
        pca9685pw = PulseWaveModulatorPCA9685PW(
            bus=SMBus('/dev/i2c-1'),
            address=PulseWaveModulatorPCA9685PW.PCA9685PW_ADDRESS,
            frequency_hz=400
        )

        # test motor
        motor = DcMotor(
            driver=DcMotorDriverIndirectPCA9685PW(
                pca9685pw=pca9685pw,
                pwm_channel=0,
                direction_pin=CkPin.GPIO21
            ),
            speed=0
        )
        motor.start()
        motor.set_speed(50)
        time.sleep(1)
        motor.set_speed(-50)
        time.sleep(1)
        motor.set_speed(0)
        motor.stop()

    try:

        test_pi_pwm_motor()
        # test_led()
        # test_rotary_encoder_state()
        # test_wait_for_stationarity()

    except KeyboardInterrupt:
        pass

    rotary_encoder.cleanup()
    locking_serial.connection.close()
    cleanup()


if __name__ == '__main__':
    main()
