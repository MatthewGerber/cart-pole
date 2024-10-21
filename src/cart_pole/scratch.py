import time

from smbus2 import SMBus
import RPi.GPIO as gpio
from raspberry_py.gpio import CkPin, setup, cleanup
from raspberry_py.gpio.integrated_circuits import PulseWaveModulatorPCA9685PW
from raspberry_py.gpio.motors import DcMotor, DcMotorDriverIndirectPCA9685PW


def main():
    """
    Scratch.
    """

    setup()

    gpio.setup(CkPin.GPIO6, gpio.OUT)
    gpio.output(CkPin.GPIO6, gpio.LOW)

    pca9685pw = PulseWaveModulatorPCA9685PW(
        bus=SMBus('/dev/i2c-1'),
        address=PulseWaveModulatorPCA9685PW.PCA9685PW_ADDRESS,
        frequency_hz=100
    )
    motor = DcMotor(
        driver=DcMotorDriverIndirectPCA9685PW(
            pca9685pw=pca9685pw,
            pwm_channel=0,
            direction_pin=CkPin.GPIO21,
            reverse=True
        ),
        speed=0
    )
    motor.start()

    try:
        while True:
            motor.set_speed(0)
            time.sleep(1.0)
            motor.set_speed(100)
            time.sleep(0.00001)
    except KeyboardInterrupt:
        pass

    motor.set_speed(0)
    cleanup()


if __name__ == '__main__':
    main()
