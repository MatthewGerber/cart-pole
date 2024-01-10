import logging
import math
import time
from argparse import ArgumentParser
from threading import Event
from typing import List, Tuple, Any, Optional

import numpy as np
from numpy.random import RandomState
from raspberry_py.gpio import CkPin
from raspberry_py.gpio.controls import LimitSwitch
from smbus2 import SMBus

from raspberry_py.gpio.integrated_circuits import PulseWaveModulatorPCA9685PW
from raspberry_py.gpio.motors import DcMotor, DcMotorDriverIndirectPCA9685PW
from raspberry_py.gpio.sensors import RotaryEncoder
from rlai.core import MdpState, Action, Agent, Reward, Environment, MdpAgent, ContinuousMultiDimensionalAction
from rlai.core.environments.mdp import MdpEnvironment
from rlai.utils import parse_arguments


class CartPoleState(MdpState):
    """
    Cart-pole state.
    """

    def __init__(
            self,
            environment: 'CartPole',
            observation: np.ndarray,
            agent: MdpAgent,
            terminal: bool,
            truncated: bool
    ):
        """
        Initialize the state.

        :param environment: Environment.
        :param observation: Observation.
        :param agent: Agent.
        :param terminal: Whether the state is terminal, meaning the episode has terminated naturally due to the
        dynamics of the environment. For example, the natural dynamics of the environment might terminate when the agent
        reaches a predefined goal state.
        :param truncated: Whether the state is truncated, meaning the episode has ended for some reason other than the
        natural dynamics of the environment. For example, imposing an artificial time limit on an episode might cause
        the episode to end without the agent in a predefined goal state.
        """

        super().__init__(
            i=agent.pi.get_state_i(observation),
            AA=environment.actions,
            terminal=terminal,
            truncated=truncated
        )

        self.observation = observation


class CartPole(MdpEnvironment):
    """
    Cart-pole environment for the Raspberry Pi.
    """

    @classmethod
    def get_argument_parser(
            cls,
    ) -> ArgumentParser:
        """
        Parse arguments.

        :return: Argument parser.
        """

        parser = ArgumentParser(
            prog=f'{cls.__module__}.{cls.__name__}',
            parents=[super().get_argument_parser()],
            allow_abbrev=False,
            add_help=False
        )

        parser.add_argument(
            '--limit-to-limit-mm',
            type=float,
            help='The distance (mm) from the left to right limit switches.'
        )

        parser.add_argument(
            '--motor-pwm-channel',
            type=int,
            help='Pulse-wave modulation (PWM) channel to use for motor control.'
        )

        parser.add_argument(
            '--motor-pwm-direction-pin',
            type=str,
            help=(
                'GPIO pin connected to the pulse-wave modulation (PWM) direction control. This can be an enumerated '
                'name and value from either the raspberry_py.gpio.Pin class (e.g., Pin.GPIO_5) or the '
                'raspberry_py.gpio.CkPin class (e.g., CkPin.GPIO5).'
            )
        )

        parser.add_argument(
            '--motor-negative-speed-is-left',
            type=bool,
            default='true',
            action='store_false',
            help='Whether negative motor speed moves the cart to the left.'
        )

        parser.add_argument(
            '--cart-rotary-encoder-phase-a-pin',
            type=str,
            help=(
                'GPIO pin connected to the phase-a input of the cart\'s rotary encoder. This can be an enumerated '
                'name and value from either the raspberry_py.gpio.Pin class (e.g., Pin.GPIO_5) or the '
                'raspberry_py.gpio.CkPin class (e.g., CkPin.GPIO5).'
            )
        )

        parser.add_argument(
            '--cart-rotary-encoder-phase-b-pin',
            type=str,
            help=(
                'GPIO pin connected to the phase-b input of the cart\'s rotary encoder. This can be an enumerated '
                'name and value from either the raspberry_py.gpio.Pin class (e.g., Pin.GPIO_5) or the '
                'raspberry_py.gpio.CkPin class (e.g., CkPin.GPIO5).'
            )
        )

        parser.add_argument(
            '--pole-rotary-encoder-phase-a-pin',
            type=str,
            help=(
                'GPIO pin connected to the phase-a input of the pole\'s rotary encoder. This can be an enumerated '
                'name and value from either the raspberry_py.gpio.Pin class (e.g., Pin.GPIO_5) or the '
                'raspberry_py.gpio.CkPin class (e.g., CkPin.GPIO5).'
            )
        )

        parser.add_argument(
            '--pole-rotary-encoder-phase-b-pin',
            type=str,
            help=(
                'GPIO pin connected to the phase-b input of the pole\'s rotary encoder. This can be an enumerated '
                'name and value from either the raspberry_py.gpio.Pin class (e.g., Pin.GPIO_5) or the '
                'raspberry_py.gpio.CkPin class (e.g., CkPin.GPIO5).'
            )
        )

        parser.add_argument(
            '--left-limit-switch-input-pin',
            type=str,
            help=(
                'GPIO pin connected to the input pin of the left limit switch. This can be an enumerated '
                'name and value from either the raspberry_py.gpio.Pin class (e.g., Pin.GPIO_5) or the '
                'raspberry_py.gpio.CkPin class (e.g., CkPin.GPIO5).'
            )
        )

        parser.add_argument(
            '--right-limit-switch-input-pin',
            type=str,
            help=(
                'GPIO pin connected to the input pin of the right limit switch. This can be an '
                'enumerated name and value from either the raspberry_py.gpio.Pin class (e.g., Pin.GPIO_5) or the '
                'raspberry_py.gpio.CkPin class (e.g., CkPin.GPIO5).'
            )
        )

        return parser

    @classmethod
    def init_from_arguments(
            cls,
            args: List[str],
            random_state: RandomState
    ) -> Tuple[Environment, List[str]]:
        """
        Initialize an environment from arguments.

        :param args: Arguments.
        :param random_state: Random state.
        :return: 2-tuple of an environment and a list of unparsed arguments.
        """

        parsed_args, unparsed_args = parse_arguments(cls, args)

        cart_pole = cls(
            name='cart-pole',
            random_state=random_state,
            **vars(parsed_args)
        )

        return cart_pole, unparsed_args

    def __init__(
            self,
            name: str,
            random_state: RandomState,
            T: Optional[int],
            limit_to_limit_mm: float,
            motor_pwm_channel: int,
            motor_pwm_direction_pin: CkPin,
            motor_negative_speed_is_left: bool,
            cart_rotary_encoder_phase_a_pin: CkPin,
            cart_rotary_encoder_phase_b_pin: CkPin,
            pole_rotary_encoder_phase_a_pin: CkPin,
            pole_rotary_encoder_phase_b_pin: CkPin,
            left_limit_switch_input_pin: CkPin,
            right_limit_switch_input_pin: CkPin
    ):
        """
        Initialize the cart-pole environment.

        :param name: Name.
        :param random_state: Random state.
        :param T: Maximum number of steps to run, or None for no limit.
        :param limit_to_limit_mm: The distance (mm) from the left to right limit switches.
        :param motor_pwm_channel: Pulse-wave modulation (PWM) channel to use for motor control.
        :param motor_pwm_direction_pin: Motor's PWM direction pin.
        :param motor_negative_speed_is_left: Whether negative motor speeds move the cart to the left.
        :param cart_rotary_encoder_phase_a_pin: Cart rotary encoder phase-a pin.
        :param cart_rotary_encoder_phase_b_pin: Cart rotary encoder phase-b pin.
        :param pole_rotary_encoder_phase_a_pin: Pole rotary encoder phase-a pin.
        :param pole_rotary_encoder_phase_b_pin: Pole rotary encoder phase-b pin.
        :param left_limit_switch_input_pin: Left limit pin.
        :param right_limit_switch_input_pin: Right limit pin.
        """

        super().__init__(
            name=name,
            random_state=random_state,
            T=T
        )

        self.limit_to_limit_mm = limit_to_limit_mm
        self.motor_pwm_channel = motor_pwm_channel
        self.motor_pwm_direction_pin = motor_pwm_direction_pin
        self.motor_negative_speed_is_left = motor_negative_speed_is_left
        self.cart_rotary_encoder_phase_a_pin = cart_rotary_encoder_phase_a_pin
        self.cart_rotary_encoder_phase_b_pin = cart_rotary_encoder_phase_b_pin
        self.pole_rotary_encoder_phase_a_pin = pole_rotary_encoder_phase_a_pin
        self.pole_rotary_encoder_phase_b_pin = pole_rotary_encoder_phase_b_pin
        self.left_limit_switch_input_pin = left_limit_switch_input_pin
        self.right_limit_switch_input_pin = right_limit_switch_input_pin

        self.actions = [
            ContinuousMultiDimensionalAction(
                value=None,
                min_values=np.array([-5.0]),
                max_values=np.array([5.0]),
                name='speed-change'
            )
        ]

        self.cart_rotary_encoder = RotaryEncoder(
            phase_a_pin=self.cart_rotary_encoder_phase_a_pin,
            phase_b_pin=self.cart_rotary_encoder_phase_b_pin,
            phase_changes_per_rotation=2400,
            report_state=False,
            degrees_per_second_smoothing=0.75
        )

        self.pole_rotary_encoder = RotaryEncoder(
            phase_a_pin=self.pole_rotary_encoder_phase_a_pin,
            phase_b_pin=self.pole_rotary_encoder_phase_b_pin,
            phase_changes_per_rotation=2400,
            report_state=False,
            degrees_per_second_smoothing=0.75
        )
        self.pole_rotary_encoder_degrees_at_bottom: Optional[float] = None

        self.pca9685pw = PulseWaveModulatorPCA9685PW(
            bus=SMBus('/dev/i2c-1'),
            address=PulseWaveModulatorPCA9685PW.PCA9685PW_ADDRESS,
            frequency_hz=500
        )

        self.motor = DcMotor(
            driver=DcMotorDriverIndirectPCA9685PW(
                pca9685pw=self.pca9685pw,
                pwm_channel=self.motor_pwm_channel,
                direction_pin=self.motor_pwm_direction_pin,
                reverse=not self.motor_negative_speed_is_left
            ),
            speed=0
        )

        self.left_limit_switch = LimitSwitch(
            input_pin=self.left_limit_switch_input_pin,
            bounce_time_ms=5
        )
        self.left_limit_pressed = Event()
        self.left_limit_released = Event()
        self.left_limit_switch.event(lambda s: self.left_limit_event(s.is_pressed()))

        self.right_limit_switch = LimitSwitch(
            input_pin=self.right_limit_switch_input_pin,
            bounce_time_ms=5
        )
        self.right_limit_pressed = Event()
        self.right_limit_released = Event()
        self.right_limit_switch.event(lambda s: self.right_limit_event(s.is_pressed()))

        self.non_episodic_motor_speed = 5

        # calibration values to be determined
        self.midline_mm = self.limit_to_limit_mm / 2.0
        self.left_limit_degrees: Optional[int] = None
        self.right_limit_degrees: Optional[int] = None
        self.limit_to_limit_degrees: Optional[float] = None
        self.cart_mm_per_degree: Optional[float] = None
        self.cart_degrees_per_mm: Optional[float] = None
        self.midline_degrees: Optional[float] = None

    def move_cart_to_left_limit(
            self
    ):
        """
        Move the cart to the left limit.
        """

        if not self.left_limit_pressed.is_set():
            self.motor.set_speed(-self.non_episodic_motor_speed)
            logging.debug('Moving cart to the left and waiting for limit switch.')
            self.left_limit_pressed.wait()

        logging.debug('Moving cart away from left limit switch.')
        self.motor.set_speed(math.ceil(self.non_episodic_motor_speed / 5.0))
        self.left_limit_released.wait()
        self.motor.set_speed(0)

    def left_limit_event(
            self,
            is_pressed: bool
    ):
        """
        Receive an event from the left limit switch.

        :param is_pressed: Whether the limit switch is pressed.
        """

        if is_pressed:
            self.left_limit_released.clear()
            self.left_limit_pressed.set()
            logging.debug('Left limit pressed.')
        else:
            self.left_limit_pressed.clear()
            self.left_limit_released.set()
            logging.debug('Left limit released.')

    def move_cart_to_right_limit(
            self
    ):
        """
        Move the cart to the right limit.
        """

        if not self.right_limit_pressed.is_set():
            logging.debug('Moving cart to the right and waiting for limit switch.')
            self.motor.set_speed(self.non_episodic_motor_speed)
            self.right_limit_pressed.wait()

        logging.debug('Moving cart away from left limit switch.')
        self.motor.set_speed(-math.ceil(self.non_episodic_motor_speed / 5.0))
        self.right_limit_released.wait()
        self.motor.set_speed(0)

    def right_limit_event(
            self,
            is_pressed: bool
    ):
        """
        Receive an event from the right limit switch.

        :param is_pressed: Whether the limit switch is pressed.
        """

        if is_pressed:
            self.right_limit_released.clear()
            self.right_limit_pressed.set()
            logging.debug('Right limit pressed.')
        else:
            self.right_limit_pressed.clear()
            self.right_limit_released.set()
            logging.debug('Right limit released.')

    def calibrate(
            self
    ):
        """
        Calibrate the cart-pole apparatus.
        """

        logging.debug('Calibrating.')

        self.move_cart_to_left_limit()
        self.left_limit_degrees = self.cart_rotary_encoder.net_total_degrees

        self.move_cart_to_right_limit()
        self.right_limit_degrees = self.cart_rotary_encoder.net_total_degrees

        self.limit_to_limit_degrees = abs(self.left_limit_degrees - self.right_limit_degrees)
        self.cart_mm_per_degree = self.limit_to_limit_mm / self.limit_to_limit_degrees
        self.cart_degrees_per_mm = self.limit_to_limit_degrees / self.limit_to_limit_mm
        self.midline_degrees = (self.left_limit_degrees + self.right_limit_degrees) / 2.0

        logging.debug(
            f'Calibrated:\n'
            f'\tLimit to limit degrees:  {self.limit_to_limit_degrees}\n'
            f'\tCart mm/deg:  {self.cart_mm_per_degree}\n'
            f'\tCart deg/mm:  {self.cart_degrees_per_mm}\n'
            f'\tMidline deg:  {self.midline_degrees}\n'
        )

    def center_cart(
            self
    ):
        """
        Center the cart.
        """

        logging.debug('Centering cart.')

        self.move_cart_to_left_limit()
        self.cart_rotary_encoder.report_state = True
        self.motor.set_speed(self.non_episodic_motor_speed)
        self.cart_rotary_encoder.event(lambda s: (
            self.motor.set_speed(0) if (
                abs(s.net_total_degrees - self.left_limit_degrees) / self.cart_degrees_per_mm >= self.midline_mm
            )
            else None
        ))
        self.cart_rotary_encoder.report_state = False

        logging.debug('Cart centered.\n')

    def wait_for_stationary_pole(
            self
    ):
        """
        Wait for the pole to become stationary.
        """

        logging.debug('Waiting for stationary pole.')

        previous_pole_num_phase_changes = self.pole_rotary_encoder.num_phase_changes
        time.sleep(1.0)
        while self.pole_rotary_encoder.num_phase_changes != previous_pole_num_phase_changes:
            time.sleep(1.0)
            logging.debug('Waiting for stationary pole.')

        self.pole_rotary_encoder_degrees_at_bottom = self.pole_rotary_encoder.net_total_degrees

        logging.debug(f'Pole is stationary at degrees:  {self.pole_rotary_encoder_degrees_at_bottom}\n')

    def reset_for_new_run(
            self,
            agent: Any
    ) -> MdpState:
        """
        Reset the environment to a random nonterminal state, if any are specified, or to None.

        :param agent: Agent used to generate on-the-fly state identifiers.
        :return: Initial state.
        """

        self.center_cart()
        self.cart_rotary_encoder.degrees_per_second = 0.0
        self.cart_rotary_encoder.update_state()
        self.wait_for_stationary_pole()
        self.pole_rotary_encoder.degrees_per_second = 0.0
        self.pole_rotary_encoder.update_state()

        return self.get_state(agent)

    def advance(
            self,
            state: MdpState,
            t: int,
            a: Action,
            agent: Agent
    ) -> Tuple[MdpState, Reward]:
        """
        Advance the environment.

        :param state: Current state.
        :param t: Current time step.
        :param a: Action.
        :param agent: Agent.
        :return: 2-tuple of the new state and a reward.
        """

        assert isinstance(a, ContinuousMultiDimensionalAction)
        assert a.value.shape == (1,)
        speed_change = round(float(a.value[0]))
        self.motor.set_speed(self.motor.get_speed() + speed_change)
        self.state = self.get_state(agent)
        reward = 1.0

        return self.state, Reward(None, reward)

    def get_state(
            self,
            agent: Any
    ) -> CartPoleState:
        """
        Get the current state.

        :param agent: Agent.
        :return: State.
        """

        return CartPoleState(
            environment=self,
            observation=np.array([
                (self.cart_rotary_encoder.net_total_degrees - self.left_limit_degrees) * self.cart_mm_per_degree,
                self.cart_rotary_encoder.degrees_per_second * self.cart_mm_per_degree,
                (self.pole_rotary_encoder.net_total_degrees - self.pole_rotary_encoder_degrees_at_bottom) % 360.0,
                self.pole_rotary_encoder.degrees_per_second
            ]),
            agent=agent,
            terminal=False,
            truncated=False
        )
