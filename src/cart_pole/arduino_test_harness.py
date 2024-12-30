import time

import serial
from serial import Serial

from raspberry_py.gpio import setup, cleanup
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
    locking_serial.connection.close()
    locking_serial.connection.open()

    x = 1.0
    while True:
        with locking_serial.lock:
            locking_serial.connection.write(RotaryEncoder.Arduino.get_bytes(x))
            x = RotaryEncoder.Arduino.get_float(locking_serial.connection.read(4))
            print(f'{x}')
            time.sleep(1.0)

    arduino_interface = RotaryEncoder.Arduino(
        phase_a_pin=3,
        phase_b_pin=5,
        phase_changes_per_rotation=1200,
        phase_change_mode=RotaryEncoder.PhaseChangeMode.ONE_SIGNAL_TWO_EDGE,
        angular_velocity_step_size=1.0,
        angular_acceleration_step_size=1.0,
        serial=locking_serial,
        identifier=1,
        state_update_hz=20
    )
    rotary_encoder = RotaryEncoder(
        interface=arduino_interface
    )
    rotary_encoder.start()
    try:
        while True:
            time.sleep(1.0 / arduino_interface.state_update_hz)
            rotary_encoder.update_state()
            state: RotaryEncoder.State = rotary_encoder.get_state()
            print(
                f'Net total degrees:  {state.net_total_degrees}\n'
                f'Degrees:  {state.degrees}\n'
                f'Clockwise:  {state.clockwise}\n'
                f'Velocity:  {state.angular_velocity} deg/s\n'
                f'Acceleration:  {state.angular_acceleration} deg/s^2\n'
            )
    except KeyboardInterrupt:
        pass
    rotary_encoder.cleanup()
    locking_serial.connection.close()
    cleanup()


if __name__ == '__main__':
    main()
