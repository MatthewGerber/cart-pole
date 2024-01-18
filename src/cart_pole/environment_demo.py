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

    def __init__(self):

        super().__init__('test', RandomState(12345), DummyPolicy(), 1.0)

        self.increment = 1.0
        self.curr_motor_speed = 0
        self.motor_speed_state_speeds: List[Tuple[int, float]] = []

    def reset_for_new_run(
            self,
            state: State
    ):
        super().reset_for_new_run(state)

        if len(self.motor_speed_state_speeds) > 0:
            df = pd.DataFrame(
                data=self.motor_speed_state_speeds,
                columns=['motor_speed', 'state_speed']
            )
            df.boxplot('state_speed', by='motor_speed', figsize=(8.0, 8.0))
            plt.xticks(rotation=45, ha='right')
            plt.xlabel('Motor speed [-100,100] (unitless)')
            plt.ylabel('Rotary encoder velocity (mm/sec)')
            plt.tight_layout()
            plt.show()

        self.curr_motor_speed = 0
        self.motor_speed_state_speeds.clear()
        self.increment *= -1.0

    def __act__(self, t: int) -> Action:

        motor_speed_increment = self.increment if t < 50 else 0.0
        self.curr_motor_speed += motor_speed_increment

        return ContinuousMultiDimensionalAction(
            value=np.array([motor_speed_increment]),
            min_values=None,
            max_values=None
        )

    def sense(
            self,
            state: State,
            t: int
    ):
        assert isinstance(state, CartPoleState)

        self.motor_speed_state_speeds.append((self.curr_motor_speed, float(state.observation[1])))


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
