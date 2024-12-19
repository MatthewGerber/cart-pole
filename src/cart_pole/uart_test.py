import time

import serial


def main():

    ser = serial.Serial(

        # Serial Port to read the data from
        port='/dev/ttyS0',

        # Rate at which the information is shared to the communication channel
        baudrate=9600,

        # Applying Parity Checking (none in this case)
        parity=serial.PARITY_NONE,

        # Pattern of Bits to be read
        stopbits=serial.STOPBITS_ONE,

        # Total number of bits to be read
        bytesize=serial.EIGHTBITS
    )

    x = 1
    while True:
        ser.write(x.to_bytes(4))
        x = int.from_bytes(ser.read(4))
        print(f'Received:  {x}')
        time.sleep(0.1)


if __name__ == '__main__':
    main()
