import time

from raspberry_py.gpio import setup, cleanup, CkPin
from raspberry_py.gpio.lights import LED


def main():
    """
    Example of using a dual multiprocess rotary encoder.
    """

    setup()

    try:
        pass

    except KeyboardInterrupt:
        pass

    cleanup()


if __name__ == '__main__':
    main()
