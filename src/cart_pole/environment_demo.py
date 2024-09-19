import logging
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.random import RandomState

from cart_pole.environment import CartPole, CartPoleState
from raspberry_py.gpio import CkPin, setup, cleanup
from rlai.core import (
    MdpAgent,
    Action,
    Monitor,
    Policy,
    MdpState,
    ContinuousMultiDimensionalAction,
    Environment,
    Agent,
    State
)


class DummyPolicy(Policy):
    """
    Dummy policy.
    """

    def __contains__(self, state: MdpState) -> bool:
        """
        Contains.

        :param state: State.
        :return: True/false.
        """

        return True

    def __getitem__(self, state: MdpState) -> Dict[Action, float]:
        """
        Get.

        :param state: State.
        :return: Action/probability.
        """

        return {}


class TestAgent(MdpAgent):
    """
    Test agent.
    """

    @classmethod
    def init_from_arguments(
            cls,
            args: List[str],
            random_state: RandomState,
            environment: Environment
    ) -> Tuple[List[Agent], List[str]]:
        """
        Init.

        :param args: Args.
        :param random_state: Randon.
        :param environment: Environment.
        :return: Agent.
        """

        raise ValueError('Cannot construct via CLI.')

    def __init__(self):
        """
        Init.
        """

        super().__init__('test', RandomState(12345), DummyPolicy(), 1.0)

        self.increment = 1
        self.curr_motor_speed = 0
        self.motor_speed_state_speeds: List[Tuple[int, float]] = []

    def reset_for_new_run(
            self,
            state: State
    ):
        """
        Reset.

        :param state: State.
        """

        super().reset_for_new_run(state)

        if len(self.motor_speed_state_speeds) > 0:
            df = pd.DataFrame(
                data=self.motor_speed_state_speeds,
                columns=['motor_speed', 'cart_velocity']
            )
            df.boxplot('cart_velocity', by='motor_speed', figsize=(8.0, 8.0))
            ticks, labels = plt.xticks()
            plt.xticks(
                ticks,  # type: ignore[arg-type]
                [label.get_text() if int(float(label.get_text())) % 5 == 0 else '' for label in labels]
            )
            plt.xticks(rotation=45, ha='right')
            plt.xlabel('Motor speed [-100,100] (unitless)')
            plt.ylabel('Cart velocity (mm/sec)')
            plt.tight_layout()
            plt.show()

        self.curr_motor_speed = 0
        self.increment *= -1

    def __act__(self, t: int) -> Action:
        """
        Get action.

        :param t: Time step.
        :return: Action.
        """

        self.curr_motor_speed += self.increment

        return ContinuousMultiDimensionalAction(
            value=np.array([self.increment]),
            min_values=None,
            max_values=None
        )

    def sense(
            self,
            state: State,
            t: int
    ):
        """
        Sense.

        :param state: State.
        :param t: Time step.
        """

        assert isinstance(state, CartPoleState)

        self.motor_speed_state_speeds.append((self.curr_motor_speed, float(state.cart_velocity_mm_per_second)))


def main():
    """
    Demonstrate the cart-pole environment.
    """

    logging.getLogger().setLevel(logging.DEBUG)

    setup()

    env = CartPole(
        name='test',
        random_state=RandomState(12345),
        T=None,
        limit_to_limit_mm=914.0,
        soft_limit_standoff_mm=100.0,
        cart_width_mm=45.0,
        motor_pwm_channel=0,
        motor_pwm_direction_pin=CkPin.GPIO21,
        motor_negative_speed_is_right=True,
        cart_rotary_encoder_phase_a_pin=CkPin.GPIO22,
        pole_rotary_encoder_speed_phase_a_pin=CkPin.GPIO5,
        pole_rotary_encoder_direction_phase_a_pin=CkPin.GPIO17,
        pole_rotary_encoder_direction_phase_b_pin=CkPin.GPIO27,
        left_limit_switch_input_pin=CkPin.GPIO20,
        right_limit_switch_input_pin=CkPin.GPIO16,
        timesteps_per_second=10.0,
        calibration_path=None,
        balance_phase_led_pin=CkPin.GPIO26,
        falling_led_pin=CkPin.GPIO19,
        termination_led_pin=CkPin.GPIO13,
        balance_gamma=0.5,
        failsafe_pwm_off_pin=CkPin.GPIO6,
        center_urf_trigger_pin=CkPin.GPIO23,
        center_urf_echo_pin=CkPin.GPIO24
    )

    agent = TestAgent()
    for _ in range(20):
        initial_state = env.reset_for_new_run(agent)
        agent.reset_for_new_run(initial_state)
        env.run(agent, Monitor())

    env.close()

    cleanup()


if __name__ == '__main__':
    main()
