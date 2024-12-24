import time

import serial

from raspberry_py.utils import IncrementalSampleAverager


def main():

    ser = serial.Serial(

        # Serial Port to read the data from
        port='/dev/ttyS0',

        # Rate at which the information is shared to the communication channel
        baudrate=115200,

        # Applying Parity Checking (none in this case)
        parity=serial.PARITY_NONE,

        # Pattern of Bits to be read
        stopbits=serial.STOPBITS_ONE,

        # Total number of bits to be read
        bytesize=serial.EIGHTBITS
    )

    x = 1
    avg_time = IncrementalSampleAverager(0, alpha=0.5)
    i = 0
    while True:
        start = time.time()
        ser.write(x.to_bytes(4))
        x = int.from_bytes(ser.read(4))
        end = time.time()
        avg_time.update(end - start)
        i += 1
        if i % 10000 == 0:
            print(f'Average time:  {avg_time.get_value()}, value {x}')


if __name__ == '__main__':
    main()
