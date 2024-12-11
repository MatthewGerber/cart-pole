import time

from raspberry_py.gpio import CkPin, setup, cleanup
from raspberry_py.gpio.sensors import MultiprocessRotaryEncoder, DualMultiprocessRotaryEncoder


def main():
    """
    Example of using a dual multiprocess rotary encoder.
    """

    setup()

    encoder = DualMultiprocessRotaryEncoder(
        CkPin.SCLK,
        CkPin.GPIO22,
        CkPin.GPIO4,
        1200,
        1.0,
        1.0
    )
    encoder.wait_for_startup()

    try:
        while True:
            time.sleep(1.0/10.0)
            encoder.update_state(True)
            state: MultiprocessRotaryEncoder.State = encoder.state
            print(
                f'Degrees:  {state.degrees}; RPM:  {60.0 * state.angular_velocity / 360.0:.1f} '
                f'(clockwise={state.clockwise})'
            )
    except KeyboardInterrupt:
        encoder.wait_for_termination()
        time.sleep(1.0)

    cleanup()


if __name__ == '__main__':
    main()
