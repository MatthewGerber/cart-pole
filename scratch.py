import time

import RPi.GPIO as gpio

from raspberry_py.gpio import CkPin, setup


def main():
    """
    Scratch file.
    """

    setup()
    gpio.setup(CkPin.GPIO24, gpio.OUT)
    while True:
        gpio.output(CkPin.GPIO24, gpio.HIGH)
        time.sleep(1.0)
        gpio.output(CkPin.GPIO24, gpio.LOW)
        time.sleep(1.0)


if __name__ == '__main__':
    main()
