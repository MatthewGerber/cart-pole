from cart_pole.environment import CartPole


def main():
    """
    Demonstrate the cart-pole environment.
    """

    env = CartPole(
        name='test',
        random_state=RandomState(12345),
        T=None,
        limit_to_limit_distance_mm=,
        inside_limits_motor_pwm_channel=0,
        outside_limits_motor_pwm_channel=1,
        motor_pwm_direction_pin=CkPin.,
        motor_negative_speed_is_left=,
        cart_rotary_encoder_phase_a_pin=CkPin.,
        cart_rotary_encoder_phase_b_pin=CkPin.,
        pole_rotary_encoder_phase_a_pin=CkPin.,
        pole_rotary_encoder_phase_b_pin=CkPin.,
        motor_side_limit_switch_input_pin=CkPin.,
        rotary_encoder_side_limit_switch_input_pin=CkPin.
    )

    env.calibrate()
    env.center_cart()


if __name__ == '__main__':
    main()
