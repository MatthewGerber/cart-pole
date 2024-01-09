from numpy.random import RandomState
from raspberry_py.gpio import CkPin

from cart_pole.environment import CartPole


def main():
    """
    Demonstrate the cart-pole environment.
    """

    env = CartPole(
        name='test',
        random_state=RandomState(12345),
        T=None,
        limit_to_limit_mm=990.0,
        motor_pwm_channel=0,
        motor_pwm_direction_pin=CkPin.GPIO21,
        motor_negative_speed_is_left=True,
        cart_rotary_encoder_phase_a_pin=CkPin.GPIO22,
        cart_rotary_encoder_phase_b_pin=CkPin.GPIO26,
        pole_rotary_encoder_phase_a_pin=CkPin.GPIO17,
        pole_rotary_encoder_phase_b_pin=CkPin.GPIO27,
        left_limit_switch_input_pin=CkPin.GPIO20,
        right_limit_switch_input_pin=CkPin.GPIO16
    )

    env.calibrate()
    env.center_cart()


if __name__ == '__main__':
    main()
