import struct
import time

import serial
from serial import Serial


def main():

    ser = Serial(
        port='/dev/serial0',
        baudrate=9600,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS
    )

    x = 1.0
    while True:
        ser.write(struct.pack('f', x))
        state_bytes = ser.read(4)
        x = struct.unpack('f', state_bytes[0:4])[0]
        print(f'{x}')
        time.sleep(1.0)


if __name__ == '__main__':
    main()
