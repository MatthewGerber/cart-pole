import time

import RPi.GPIO as gpio

from raspberry_py.gpio import CkPin, setup


def main():
    setup()
    pin = CkPin.RXD0
    gpio.setup(pin, gpio.OUT)
    gpio.output(pin, gpio.HIGH)
    time.sleep(1.0)
    gpio.output(pin, gpio.LOW)


if __name__ == '__main__':
    main()
