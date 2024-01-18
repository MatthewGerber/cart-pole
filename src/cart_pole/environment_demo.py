import logging
from typing import List, Tuple, Dict

import numpy as np
from numpy.random import RandomState

from cart_pole.environment import CartPole
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

    def __contains__(self, state: MdpState) -> bool:
        return True

    def __getitem__(self, state: MdpState) -> Dict[Action, float]:
        return {}


class TestAgent(MdpAgent):

    @classmethod
    def init_from_arguments(
            cls,
            args: List[str],
            random_state: RandomState,
            environment: Environment
    ) -> Tuple[List[Agent], List[str]]:
        pass

    def reset_for_new_run(
            self,
            state: State
    ):
        super().reset_for_new_run(state)

        self.increment *= -1.0

    def __init__(self):

        super().__init__('test', RandomState(12345), DummyPolicy(), 1.0)

        self.increment = 1.0

    def __act__(self, t: int) -> Action:

        return ContinuousMultiDimensionalAction(
            value=np.array([self.increment if t < 50 else 0.0]),
            min_values=None,
            max_values=None
        )


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
        soft_limit_standoff=100.0,
        cart_width_mm=45.0,
        motor_pwm_channel=0,
        motor_pwm_direction_pin=CkPin.GPIO21,
        motor_negative_speed_is_left=False,
        cart_rotary_encoder_phase_a_pin=CkPin.GPIO22,
        cart_rotary_encoder_phase_b_pin=CkPin.GPIO26,
        pole_rotary_encoder_phase_a_pin=CkPin.GPIO17,
        pole_rotary_encoder_phase_b_pin=CkPin.GPIO27,
        left_limit_switch_input_pin=CkPin.GPIO20,
        right_limit_switch_input_pin=CkPin.GPIO16,
        max_timesteps_per_second=20.0
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
