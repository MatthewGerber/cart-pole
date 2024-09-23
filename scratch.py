import time

import RPi.GPIO as gpio

from raspberry_py.gpio import CkPin, setup


def main():
    """
    Scratch file.
    """

    setup()
    gpio.setup(CkPin.MISO, gpio.OUT)
    while True:
        gpio.output(CkPin.MISO, gpio.HIGH)
        time.sleep(1.0)
        gpio.output(CkPin.MISO, gpio.LOW)
        time.sleep(1.0)


if __name__ == '__main__':
    main()
