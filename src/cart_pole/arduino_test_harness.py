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

    while True:
        ser.write((42).to_bytes(4))
        state_bytes = ser.read(8)
        cart_index = int.from_bytes(state_bytes[0:4], signed=True)
        pole_index = int.from_bytes(state_bytes[4:8], signed=True)
        print(f'cart index:  {cart_index}, pole index:  {pole_index}')
        time.sleep(0.01)


if __name__ == '__main__':
    main()
