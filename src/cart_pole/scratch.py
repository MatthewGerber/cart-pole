import time

from raspberry_py.gpio import CkPin, setup, cleanup
from raspberry_py.gpio.lights import LED


def main():
    """
    Scratch.
    """

    setup()
    leds = [
        LED(CkPin.GPIO26),
        LED(CkPin.GPIO19),
        LED(CkPin.GPIO13)
    ]
    for _ in range(10):
        for led in leds:
            led.turn_on()
            time.sleep(0.01)
            led.turn_off()
            time.sleep(0.01)
    cleanup()


if __name__ == '__main__':
    main()
