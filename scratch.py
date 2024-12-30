import time

from raspberry_py.gpio import setup, cleanup, CkPin
from raspberry_py.gpio.lights import LED


def main():
    """
    Example of using a dual multiprocess rotary encoder.
    """

    setup()

    try:
        led = LED(CkPin.GPIO19)
        while True:
            time.sleep(1.0)
            if led.is_on():
                led.turn_off()
            else:
                led.turn_on()

    except KeyboardInterrupt:
        pass

    cleanup()


if __name__ == '__main__':
    main()
