import logging
import os.path
import pickle
import time
from argparse import ArgumentParser
from datetime import timedelta
from enum import Enum, auto
from threading import Event, RLock
from typing import List, Tuple, Any, Optional, Dict

import RPi.GPIO as gpio
import numpy as np
from numpy.random import RandomState
from smbus2 import SMBus

from raspberry_py.gpio import CkPin, get_ck_pin, setup, cleanup
from raspberry_py.gpio.controls import LimitSwitch
from raspberry_py.gpio.integrated_circuits import PulseWaveModulatorPCA9685PW
from raspberry_py.gpio.lights import LED
from raspberry_py.gpio.motors import DcMotor, DcMotorDriverIndirectPCA9685PW
from raspberry_py.gpio.sensors import (
    MultiprocessRotaryEncoder,
    RotaryEncoder,
    DualMultiprocessRotaryEncoder,
    UltrasonicRangeFinder
)
from rlai.core import MdpState, Action, Agent, Reward, Environment, MdpAgent, ContinuousMultiDimensionalAction
from rlai.core.environments.mdp import ContinuousMdpEnvironment
from rlai.utils import parse_arguments, IncrementalSampleAverager


class CartPoleAction(ContinuousMultiDimensionalAction):
    """
    Cart-pole action.
    """

    def __init__(
            self,
            value: Optional[np.ndarray],
            min_values: Optional[np.ndarray],
            max_values: Optional[np.ndarray],
            name: Optional[str] = None
    ):
        """
        Initialize the action.

        :param value: Value.
        :param min_values: Minimum values.
        :param max_values: Maximum values.
        :param name: Name.
        """

        super().__init__(
            value=value,
            min_values=min_values,
            max_values=max_values,
            name=name
        )

    def __eq__(
            self,
            other
    ) -> bool:
        """
        Check equality.

        :param other: Other object.
        """

        if not isinstance(other, CartPoleAction):
            raise ValueError(f'Expected a {CartPoleAction}')

        return np.allclose(self.value, other.value, atol=0.0001)

    def __hash__(self) -> int:
        """
        Get hash.

        :return: Hash.
        """

        return super().__hash__()


class EpisodePhase(Enum):
    """
    Episode phases.
    """

    # The initial phase, in which the cart must oscillate left and right to build angular momentum that swings the
    # pole to the vertical position.
    SWING_UP = auto()

    # The pole is upright with respect to a progressive threshold.
    PROGRESSIVE_UPRIGHT = auto()

    # The pole is balancing properly with respect to angle and velocity.
    BALANCE = auto()


class CartPosition(Enum):
    """
    Cart position.
    """

    # Cart is left of center.
    LEFT_OF_CENTER = auto()

    # Cart is centered exactly.
    CENTERED = auto()

    # Cart is right of center.
    RIGHT_OF_CENTER = auto()


class CartDirection(Enum):
    """
    Cart direction.
    """

    # Cart is moving to the left.
    LEFT = auto()

    # Cart is moving to the right.
    RIGHT = auto()


class CartRotaryEncoder(MultiprocessRotaryEncoder):
    """
    Extension of the multiprocess rotary encoder that adds cart centering.
    """

    def wait_for_cart_to_cross_center(
            self,
            original_position: CartPosition,
            left_limit_degrees: float,
            cart_mm_per_degree: float,
            midline_mm: float,
            check_delay_seconds: float
    ):
        """
        Wait for cart to cross the center.

        :param original_position: Original position.
        :param left_limit_degrees: Left limit degrees.
        :param cart_mm_per_degree: Cart mm/degree.
        :param midline_mm: Midline mm.
        :param check_delay_seconds: Check delay seconds.
        """

        while CartPole.get_cart_position(
            cart_net_total_degrees=self.get_net_total_degrees(True),
            left_limit_degrees=left_limit_degrees,
            cart_mm_per_degree=cart_mm_per_degree,
            midline_mm=midline_mm
        ) == original_position:
            time.sleep(check_delay_seconds)


class CartPoleState(MdpState):
    """
    Cart-pole state.
    """

    class Dimension(Enum):
        """
        Dimensions.
        """

        CartPosition = 0
        CartVelocity = 1
        PoleAngle = 2
        PoleVelocity = 3
        PoleAcceleration = 4

    @staticmethod
    def zero_to_one_pole_angle_from_degrees(
            degrees_from_upright: float
    ) -> float:
        """
        Get [0.0, 1.0] pole angle, with 0.0 being straight down (the worst) and 1.0 being straight up (the best). Useful
        in reward calculations.

        :param degrees_from_upright: Degrees from upright.
        :return: Pole angle in [0.0, 1.0].
        """

        return (180.0 - abs(degrees_from_upright)) / 180.0

    def __init__(
            self,
            environment: 'CartPole',
            cart_mm_from_center: float,
            cart_velocity_mm_per_sec: float,
            pole_angle_deg_from_upright: float,
            pole_angular_velocity_deg_per_sec: float,
            pole_angular_acceleration_deg_per_sec_squared: float,
            step: Optional[int],
            agent: MdpAgent,
            terminal: bool,
            truncated: bool
    ):
        """
        Initialize the state.

        :param environment: Environment.
        :param cart_mm_from_center: Cart position as mm left of (negative), right of (positive), or at (zero) center.
        :param cart_velocity_mm_per_sec: Cart velocity (mm/sec).
        :param pole_angle_deg_from_upright: Pole angle as degrees left of (negative), right of (positive), or at (zero)
        upright.
        :param pole_angular_velocity_deg_per_sec: Pole angular velocity (deg/sec).
        :param pole_angular_acceleration_deg_per_sec_squared: Pole angular acceleration (deg/sec^2).
        :param step: Time step.
        :param agent: Agent.
        :param terminal: Whether the state is terminal, meaning the episode has terminated naturally due to the
        dynamics of the environment. For example, the natural dynamics of the environment terminate when the cart goes
        beyond the permitted limits of the track.
        :param truncated: Whether the state is truncated, meaning the episode has ended for some reason other than the
        natural dynamics of the environment. For example, imposing an artificial time limit on an episode might cause
        the episode to end without the agent in a terminal state.
        """

        self.cart_mm_from_center = cart_mm_from_center
        self.cart_velocity_mm_per_second = cart_velocity_mm_per_sec
        self.pole_angle_deg_from_upright = pole_angle_deg_from_upright
        self.pole_angular_velocity_deg_per_sec = pole_angular_velocity_deg_per_sec
        self.pole_angular_acceleration_deg_per_sec_squared = pole_angular_acceleration_deg_per_sec_squared
        self.step = step

        self.observation = np.array([
            self.cart_mm_from_center,
            self.cart_velocity_mm_per_second,
            self.pole_angle_deg_from_upright,
            self.pole_angular_velocity_deg_per_sec,
            self.pole_angular_acceleration_deg_per_sec_squared
        ])

        # pole angle in [0.0, 1.0] where 0.0 is straight down, and 1.0 is straight up.
        self.zero_to_one_pole_angle = CartPoleState.zero_to_one_pole_angle_from_degrees(
            self.pole_angle_deg_from_upright
        )

        # pole angular speed in [0.0, 1.0] where 0.0 is full speed, and 1.0 is stationary.
        self.zero_to_one_pole_angular_speed = 1.0 - min(
            1.0,
            abs(self.pole_angular_velocity_deg_per_sec) / environment.max_pole_angular_speed_deg_per_second
        )

        # pole angular acceleration in [0.0, 1.0] where 0.0 is full acceleration, and 1.0 is no acceleration.
        self.zero_to_one_pole_angular_acceleration = 1.0 - min(
            1.0,
            (
                abs(self.pole_angular_acceleration_deg_per_sec_squared) /
                environment.max_pole_angular_acceleration_deg_per_second_squared
            )
        )

        # evaluate the pole angle a small fraction of a second (0.00001) from the current time to determine whether it
        # is falling. if we look too far ahead, we'll go beyond the point where the pole is vertical and the calculation
        # will provide the wrong answer.
        self.pole_is_falling = self.zero_to_one_pole_angle > self.zero_to_one_pole_angle_from_degrees(
            self.pole_angle_deg_from_upright + (self.pole_angular_velocity_deg_per_sec * 0.00001)
        )

        # cart distance from center in [0.0, 1.0] where 0.0 is either side's soft limit, and 1.0 is centered.
        self.zero_to_one_cart_distance_from_center = 1.0 - min(
            1.0,
            abs(self.cart_mm_from_center) / environment.soft_limit_mm_from_midline
        )

        # cart speed in [0.0, 1.0] where 0.0 is full speed, and 1.0 is stationary.
        self.zero_to_one_cart_speed = 1.0 - min(
            1.0,
            abs(self.cart_velocity_mm_per_second) / environment.max_cart_speed_mm_per_second
        )

        super().__init__(
            i=agent.pi.get_state_i(self.observation),
            AA=environment.actions,
            terminal=terminal,
            truncated=truncated
        )

    def __str__(
            self
    ) -> str:
        """
        Get string.

        :return: String.
        """

        return (
            f'cart pos={self.cart_mm_from_center:.1f} mm; 0-1 pos={self.zero_to_one_cart_distance_from_center:.2f}; '
            f'vel={self.cart_velocity_mm_per_second:.1f} mm/s; ' 
            f'pole pos={self.pole_angle_deg_from_upright:.1f} deg; 0-1 pos={self.zero_to_one_pole_angle:.2f}; '
            f'falling={self.pole_is_falling} @ {self.pole_angular_velocity_deg_per_sec:.1f} deg/s'
        )

    def __eq__(
            self,
            other
    ) -> bool:
        """
        Check equality.

        :param other: Other object.
        """

        if not isinstance(other, CartPoleState):
            raise ValueError(f'Expected a {CartPoleState}')

        return np.allclose(self.observation, other.observation, atol=0.0001)


class CartPole(ContinuousMdpEnvironment):
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
            '--soft-limit-standoff-mm',
            type=float,
            help='Soft-limit standoff distance (mm) to maintain from the hard limits.'
        )

        parser.add_argument(
            '--cart-width-mm',
            type=float,
            help='Width (mm) of the cart that hits the limits.'
        )

        parser.add_argument(
            '--motor-pwm-channel',
            type=int,
            help='Pulse-wave modulation (PWM) channel to use for motor control.'
        )

        parser.add_argument(
            '--motor-pwm-direction-pin',
            type=get_ck_pin,
            help=(
                'GPIO pin connected to the pulse-wave modulation (PWM) direction control. This can be an enumerated '
                'type and name from either the raspberry_py.gpio.Pin class (e.g., Pin.GPIO_5) or the '
                'raspberry_py.gpio.CkPin class (e.g., CkPin.GPIO5).'
            )
        )

        parser.add_argument(
            '--motor-negative-speed-is-right',
            default='false',
            action='store_true',
            help='Whether negative motor speed moves the cart to the right.'
        )

        parser.add_argument(
            '--cart-rotary-encoder-phase-a-pin',
            type=get_ck_pin,
            help=(
                'GPIO pin connected to the phase-a input of the cart\'s rotary encoder. This can be an enumerated '
                'type and name from either the raspberry_py.gpio.Pin class (e.g., Pin.GPIO_5) or the '
                'raspberry_py.gpio.CkPin class (e.g., CkPin.GPIO5).'
            )
        )

        parser.add_argument(
            '--pole-rotary-encoder-speed-phase-a-pin',
            type=get_ck_pin,
            help=(
                'GPIO pin connected to the phase-a input of the pole\'s rotary encoder. This can be an enumerated '
                'type and name from either the raspberry_py.gpio.Pin class (e.g., Pin.GPIO_5) or the '
                'raspberry_py.gpio.CkPin class (e.g., CkPin.GPIO5).'
            )
        )

        parser.add_argument(
            '--pole-rotary-encoder-direction-phase-a-pin',
            type=get_ck_pin,
            help=(
                'GPIO pin connected to the phase-a input of the pole\'s rotary encoder. This can be an enumerated '
                'type and name from either the raspberry_py.gpio.Pin class (e.g., Pin.GPIO_5) or the '
                'raspberry_py.gpio.CkPin class (e.g., CkPin.GPIO5).'
            )
        )

        parser.add_argument(
            '--pole-rotary-encoder-direction-phase-b-pin',
            type=get_ck_pin,
            help=(
                'GPIO pin connected to the phase-b input of the pole\'s rotary encoder. This can be an enumerated '
                'type and name from either the raspberry_py.gpio.Pin class (e.g., Pin.GPIO_5) or the '
                'raspberry_py.gpio.CkPin class (e.g., CkPin.GPIO5).'
            )
        )

        parser.add_argument(
            '--left-limit-switch-input-pin',
            type=get_ck_pin,
            help=(
                'GPIO pin connected to the input pin of the left limit switch. This can be an enumerated type and name '
                'from either the raspberry_py.gpio.Pin class (e.g., Pin.GPIO_5) or the raspberry_py.gpio.CkPin class '
                '(e.g., CkPin.GPIO5).'
            )
        )

        parser.add_argument(
            '--right-limit-switch-input-pin',
            type=get_ck_pin,
            help=(
                'GPIO pin connected to the input pin of the right limit switch. This can be an enumerated type and name '
                'from either the raspberry_py.gpio.Pin class (e.g., Pin.GPIO_5) or the raspberry_py.gpio.CkPin class '
                '(e.g., CkPin.GPIO5).'
            )
        )

        parser.add_argument(
            '--timesteps-per-second',
            type=float,
            help='Number of environment advancement steps to execute per second.'
        )

        parser.add_argument(
            '--calibration-path',
            type=str,
            help=(
                'Path to calibration pickle file. A new calibration will be saved to this path if it does not exist. '
                'If the path does exist, then its calibration values will be used instead of performing a new '
                'calibration.'
            )
        )

        parser.add_argument(
            '--progressive-upright-led-pin',
            type=get_ck_pin,
            default=None,
            help=(
                'GPIO pin connected to an LED to illuminate when the pole reaches the progressive upright position. '
                'This can be an enumerated type and name from either the raspberry_py.gpio.Pin class (e.g., '
                'Pin.GPIO_5) or the raspberry_py.gpio.CkPin class (e.g., CkPin.GPIO5).'
            )
        )

        parser.add_argument(
            '--falling-led-pin',
            type=get_ck_pin,
            default=None,
            help=(
                'GPIO pin connected to an LED to illuminate when the pole is falling. This can be an enumerated type '
                'and name from either the raspberry_py.gpio.Pin class (e.g., Pin.GPIO_5) or the '
                'raspberry_py.gpio.CkPin class (e.g., CkPin.GPIO5).'
            )
        )

        parser.add_argument(
            '--cart-moving-right-led-pin',
            type=get_ck_pin,
            default=None,
            help=(
                'GPIO pin connected to an LED to illuminate when the cart is moving right. This can be an enumerated '
                'type and name from either the raspberry_py.gpio.Pin class (e.g., Pin.GPIO_5) or the '
                'raspberry_py.gpio.CkPin class (e.g., CkPin.GPIO5).'
            )
        )

        parser.add_argument(
            '--balance-led-pin',
            type=get_ck_pin,
            default=None,
            help=(
                'GPIO pin connected to an LED to illuminate when the pole is balancing. This can be an enumerated type '
                'and name from either the raspberry_py.gpio.Pin class (e.g., Pin.GPIO_5) or the '
                'raspberry_py.gpio.CkPin class (e.g., CkPin.GPIO5).'
            )
        )

        parser.add_argument(
            '--termination-led-pin',
            type=get_ck_pin,
            default=None,
            help=(
                'GPIO pin connected to an LED to illuminate when the episode terminates. This can be an enumerated '
                'type and name from either the raspberry_py.gpio.Pin class (e.g., Pin.GPIO_5) or the '
                'raspberry_py.gpio.CkPin class (e.g., CkPin.GPIO5).'
            )
        )

        parser.add_argument(
            '--balance-gamma',
            type=float,
            help='Gamma (discount) to use during the balancing phase of the episode.'
        )

        parser.add_argument(
            '--failsafe-pwm-off-pin',
            type=get_ck_pin,
            help=(
                'GPIO pin connected to the failsafe transistor that turns the motor PWM off. This can be an enumerated '
                'type and name from either the raspberry_py.gpio.Pin class (e.g., Pin.GPIO_5) or the '
                'raspberry_py.gpio.CkPin class (e.g., CkPin.GPIO5).'
            )
        )

        parser.add_argument(
            '--centering-range-finder-trigger-pin',
            type=get_ck_pin,
            help=(
                'GPIO pin connected to the trigger pin of the centering ultrasonic range finder. This can be an '
                'enumerated type and name from either the raspberry_py.gpio.Pin class (e.g., Pin.GPIO_5) or the '
                'raspberry_py.gpio.CkPin class (e.g., CkPin.GPIO5).'
            )
        )

        parser.add_argument(
            '--centering-range-finder-echo-pin',
            type=get_ck_pin,
            help=(
                'GPIO pin connected to the echo pin of the centering ultrasonic range finder. This can be an '
                'enumerated type and name from either the raspberry_py.gpio.Pin class (e.g., Pin.GPIO_5) or the '
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

    @staticmethod
    def get_cart_position(
            cart_net_total_degrees: float,
            left_limit_degrees: float,
            cart_mm_per_degree: float,
            midline_mm: float
    ) -> CartPosition:
        """
        Check whether the cart is left of the midline.

        :param cart_net_total_degrees: Cart net total degrees.
        :param left_limit_degrees: Left-limit degrees.
        :param cart_mm_per_degree: Cart mm/degree.
        :param midline_mm: Midline mm.
        :return: True if the cart is left of the midline and False otherwise.
        """

        cart_mm_from_left_limit = abs(cart_net_total_degrees - left_limit_degrees) * cart_mm_per_degree

        if cart_mm_from_left_limit < midline_mm:
            position = CartPosition.LEFT_OF_CENTER
        elif np.isclose(cart_mm_from_left_limit, midline_mm):
            position = CartPosition.CENTERED
        else:
            position = CartPosition.RIGHT_OF_CENTER

        return position

    def __init__(
            self,
            name: str,
            random_state: RandomState,
            T: Optional[int],
            limit_to_limit_mm: float,
            soft_limit_standoff_mm: float,
            cart_width_mm: float,
            motor_pwm_channel: int,
            motor_pwm_direction_pin: CkPin,
            motor_negative_speed_is_right: bool,
            cart_rotary_encoder_phase_a_pin: CkPin,
            pole_rotary_encoder_speed_phase_a_pin: CkPin,
            pole_rotary_encoder_direction_phase_a_pin: CkPin,
            pole_rotary_encoder_direction_phase_b_pin: CkPin,
            left_limit_switch_input_pin: CkPin,
            right_limit_switch_input_pin: CkPin,
            timesteps_per_second: float,
            calibration_path: Optional[str],
            progressive_upright_led_pin: Optional[CkPin],
            falling_led_pin: Optional[CkPin],
            cart_moving_right_led_pin: Optional[CkPin],
            balance_led_pin: Optional[CkPin],
            termination_led_pin: Optional[CkPin],
            balance_gamma: float,
            failsafe_pwm_off_pin: CkPin,
            centering_range_finder_trigger_pin: CkPin,
            centering_range_finder_echo_pin: CkPin
    ):
        """
        Initialize the cart-pole environment.

        :param name: Name.
        :param random_state: Random state.
        :param T: Maximum number of steps to run, or None for no limit.
        :param limit_to_limit_mm: The distance (mm) from the left to right limit switches.
        :param soft_limit_standoff_mm: Soft-limit standoff distance (mm) to maintain from the hard limits.
        :param cart_width_mm: Width (mm) of the cart that hits the limits.
        :param motor_pwm_channel: Pulse-wave modulation (PWM) channel to use for motor control.
        :param motor_pwm_direction_pin: Motor's PWM direction pin.
        :param motor_negative_speed_is_right: Whether negative motor speeds move the cart to the right.
        :param cart_rotary_encoder_phase_a_pin: Cart rotary encoder phase-a pin.
        :param pole_rotary_encoder_speed_phase_a_pin: Pole rotary encoder phase-a pin.
        :param pole_rotary_encoder_direction_phase_a_pin: Pole rotary encoder phase-a pin.
        :param pole_rotary_encoder_direction_phase_b_pin: Pole rotary encoder phase-b pin.
        :param left_limit_switch_input_pin: Left limit pin.
        :param right_limit_switch_input_pin: Right limit pin.
        :param timesteps_per_second: Number of environment advancement steps to execute per second.
        :param calibration_path: Path to calibration pickle to read/write.
        :param progressive_upright_led_pin: Pin connected to an LED to illuminate when the pole reaches the
        progressive upright angle, or pass None to ignore.
        :param falling_led_pin: Pin connected to an LED to illuminate when the pole is falling, or pass None to ignore.
        :param cart_moving_right_led_pin: Pin connected to an LED to illuminate when the cart is moving right, or pass
        None to ignore.
        :param balance_led_pin: Pin connected to an LED to illuminate when the pole is balancing properly, or pass None
        to ignore.
        :param termination_led_pin: Pin connected to an LED to illuminate when the episode terminates, or pass None to
        ignore.
        :param balance_gamma: Gamma (discount) to use during the balancing.
        :param failsafe_pwm_off_pin: Failsafe PWM off pin.
        :param centering_range_finder_trigger_pin: Trigger pin of the ultrasonic range finder at the center position.
        :param centering_range_finder_echo_pin: Echo pin of the ultrasonic range finder at the center position.
        """

        super().__init__(
            name=name,
            random_state=random_state,
            T=T
        )

        self.limit_to_limit_mm = limit_to_limit_mm
        self.soft_limit_standoff_mm = soft_limit_standoff_mm
        self.cart_width_mm = cart_width_mm
        self.motor_pwm_channel = motor_pwm_channel
        self.motor_pwm_direction_pin = motor_pwm_direction_pin
        self.motor_negative_speed_is_right = motor_negative_speed_is_right
        self.cart_rotary_encoder_phase_a_pin = cart_rotary_encoder_phase_a_pin
        self.pole_rotary_encoder_speed_phase_a_pin = pole_rotary_encoder_speed_phase_a_pin
        self.pole_rotary_encoder_direction_phase_a_pin = pole_rotary_encoder_direction_phase_a_pin
        self.pole_rotary_encoder_direction_phase_b_pin = pole_rotary_encoder_direction_phase_b_pin
        self.left_limit_switch_input_pin = left_limit_switch_input_pin
        self.right_limit_switch_input_pin = right_limit_switch_input_pin
        self.timesteps_per_second = timesteps_per_second
        self.calibration_path = os.path.expanduser(calibration_path)
        self.progressive_upright_led_pin = progressive_upright_led_pin
        self.falling_led_pin = falling_led_pin
        self.cart_moving_right_led_pin = cart_moving_right_led_pin
        self.balance_led_pin = balance_led_pin
        self.termination_led_pin = termination_led_pin
        self.balance_gamma = balance_gamma
        self.failsafe_pwm_off_pin = failsafe_pwm_off_pin
        self.centering_range_finder_trigger_pin = centering_range_finder_trigger_pin
        self.centering_range_finder_echo_pin = centering_range_finder_echo_pin

        # non-calibrated attributes
        self.midline_mm = self.limit_to_limit_mm / 2.0
        self.soft_limit_mm_from_midline = self.midline_mm - self.soft_limit_standoff_mm - self.cart_width_mm / 2.0
        self.agent: Optional[MdpAgent] = None
        self.previous_timestep_epoch: Optional[float] = None
        self.current_timesteps_per_second = IncrementalSampleAverager(initial_value=0.0, alpha=0.25)
        self.timestep_sleep_seconds = 1.0 / self.timesteps_per_second
        self.min_seconds_for_full_motor_speed_range = 0.20
        self.original_agent_gamma: Optional[float] = None
        self.truncation_gamma: Optional[float] = None  # unused. unclear if this is effective.
        self.max_pole_angular_speed_deg_per_second = 1080.0
        self.max_pole_angular_acceleration_deg_per_second_squared = 8000.0
        self.episode_phase = EpisodePhase.SWING_UP
        self.progressive_upright_pole_angle = 175.0
        self.achieved_progressive_upright = False
        self.balance_pole_angle = 35.0
        self.balance_angular_velocity = 8.0 * self.balance_pole_angle
        self.lost_balance_timestamp: Optional[float] = None
        self.lost_balance_timer_seconds = 20.0
        self.cart_rotary_encoder_angular_velocity_step_size = 0.9
        self.cart_rotary_encoder_angular_acceleration_step_size = 0.25
        self.pole_rotary_encoder_angular_velocity_step_size = 0.9
        self.pole_rotary_encoder_angular_acceleration_step_size = 0.25
        self.fraction_time_balancing = IncrementalSampleAverager()

        (
            self.state_lock,
            self.pca9685pw,
            self.motor,
            self.cart_rotary_encoder,
            self.pole_rotary_encoder,
            self.left_limit_switch,
            self.left_limit_pressed,
            self.left_limit_released,
            self.right_limit_switch,
            self.right_limit_pressed,
            self.right_limit_released,
            self.progressive_upright_led,
            self.falling_led,
            self.cart_moving_right_led,
            self.balance_led,
            self.termination_led,
            self.calibrate_on_next_reset,
            self.centering_range_finder,
            self.leds
        ) = self.get_components()

        # configure the continuous action with a single dimension ranging the maximum motor speed change
        min_timesteps_for_full_motor_speed_range = max(
            1.0,
            self.timesteps_per_second * self.min_seconds_for_full_motor_speed_range
        )
        motor_speed_range = float(self.motor.max_speed - self.motor.min_speed)
        max_motor_speed_change_per_step = motor_speed_range / min_timesteps_for_full_motor_speed_range
        self.actions = [
            ContinuousMultiDimensionalAction(
                value=None,
                min_values=np.array([-max_motor_speed_change_per_step]),
                max_values=np.array([max_motor_speed_change_per_step]),
                name='motor-speed-change'
            )
        ]

        if self.calibrate_on_next_reset:
            self.motor_deadzone_speed_left: Optional[int] = None
            self.motor_deadzone_speed_right: Optional[int] = None
            self.left_limit_degrees: Optional[float] = None
            self.right_limit_degrees: Optional[float] = None
            self.limit_to_limit_degrees: Optional[float] = None
            self.cart_mm_per_degree: Optional[float] = None
            self.midline_degrees: Optional[float] = None
            self.max_cart_speed_mm_per_second: Optional[float] = None
            self.cart_phase_change_index_at_center: Optional[int] = None
            self.cart_phase_change_index_at_left_limit: Optional[int] = None
            self.cart_phase_change_index_at_right_limit: Optional[int] = None
            self.pole_phase_change_index_at_bottom: Optional[int] = None
            self.pole_degrees_at_bottom: Optional[float] = None
            self.calibrate_on_next_reset = True

    def __getstate__(
            self
    ) -> Dict:
        """
        Get state to pickle.

        :return: State.
        """

        state = dict(self.__dict__)

        state['state_lock'] = None
        state['pca9685pw'] = None
        state['motor'] = None
        state['cart_rotary_encoder'] = None
        state['pole_rotary_encoder'] = None
        state['left_limit_switch'] = None
        state['left_limit_pressed'] = None
        state['left_limit_released'] = None
        state['right_limit_switch'] = None
        state['right_limit_pressed'] = None
        state['right_limit_released'] = None
        state['progressive_upright_led'] = None
        state['falling_led'] = None
        state['cart_moving_right_led'] = None
        state['balance_led'] = None
        state['termination_led'] = None
        state['centering_range_finder'] = None
        state['leds'] = None

        return state

    def __setstate__(
            self,
            state: Dict
    ):
        """
        Set state from pickle.

        :param state: State.
        """

        self.__dict__ = state

        (
            self.state_lock,
            self.pca9685pw,
            self.motor,
            self.cart_rotary_encoder,
            self.pole_rotary_encoder,
            self.left_limit_switch,
            self.left_limit_pressed,
            self.left_limit_released,
            self.right_limit_switch,
            self.right_limit_pressed,
            self.right_limit_released,
            self.progressive_upright_led,
            self.falling_led,
            self.cart_moving_right_led,
            self.balance_led,
            self.termination_led,
            self.calibrate_on_next_reset,
            self.centering_range_finder,
            self.leds
        ) = self.get_components()

    def get_components(
            self
    ) -> Tuple[
        RLock,
        PulseWaveModulatorPCA9685PW,
        DcMotor,
        CartRotaryEncoder,
        DualMultiprocessRotaryEncoder,
        LimitSwitch,
        Event,
        Event,
        LimitSwitch,
        Event,
        Event,
        Optional[LED],
        Optional[LED],
        Optional[LED],
        Optional[LED],
        Optional[LED],
        bool,
        UltrasonicRangeFinder,
        List[Optional[LED]]
    ]:
        """
        Get circuitry components and other attributes that cannot be pickled. This is primarily used to restore the 
        environment when resuming training.

        :return: Tuple of components.
        """

        setup()

        pca9685pw = PulseWaveModulatorPCA9685PW(
            bus=SMBus('/dev/i2c-1'),
            address=PulseWaveModulatorPCA9685PW.PCA9685PW_ADDRESS,
            frequency_hz=500
        )

        # we use the motor output speed (+/-) as our direction signal, so we can use a single-signal encoder for the
        # cart. a dual-multiprocess encoder would be better, but the pi only has four cores. use one here, two for the
        # pole below, and the fourth for the main thread of the program.
        cart_rotary_encoder = CartRotaryEncoder(
            phase_a_pin=self.cart_rotary_encoder_phase_a_pin,
            phase_b_pin=None,
            phase_changes_per_rotation=1200,
            phase_change_mode=RotaryEncoder.PhaseChangeMode.ONE_SIGNAL_ONE_EDGE,
            angular_velocity_step_size=self.cart_rotary_encoder_angular_velocity_step_size,
            angular_acceleration_step_size=self.cart_rotary_encoder_angular_acceleration_step_size
        )
        cart_rotary_encoder.wait_for_startup()

        pole_rotary_encoder = DualMultiprocessRotaryEncoder(
            speed_phase_a_pin=self.pole_rotary_encoder_speed_phase_a_pin,
            direction_phase_a_pin=self.pole_rotary_encoder_direction_phase_a_pin,
            direction_phase_b_pin=self.pole_rotary_encoder_direction_phase_b_pin,
            phase_changes_per_rotation=1200,
            angular_velocity_step_size=self.pole_rotary_encoder_angular_velocity_step_size,
            angular_acceleration_step_size=self.pole_rotary_encoder_angular_acceleration_step_size
        )
        pole_rotary_encoder.wait_for_startup()

        left_limit_switch = LimitSwitch(
            input_pin=self.left_limit_switch_input_pin,
            bounce_time_ms=5
        )
        left_limit_pressed = Event()
        left_limit_released = Event()
        if left_limit_switch.is_pressed():
            left_limit_pressed.set()
        else:
            left_limit_released.set()
        left_limit_switch.event(lambda s: self.left_limit_event(s.pressed))

        right_limit_switch = LimitSwitch(
            input_pin=self.right_limit_switch_input_pin,
            bounce_time_ms=5
        )
        right_limit_pressed = Event()
        right_limit_released = Event()
        if right_limit_switch.is_pressed():
            right_limit_pressed.set()
        else:
            right_limit_released.set()
        right_limit_switch.event(lambda s: self.right_limit_event(s.pressed))

        gpio.setup(self.failsafe_pwm_off_pin, gpio.OUT)
        self.enable_motor_pwm()

        progressive_upright_led = (
            None if self.progressive_upright_led_pin is None
            else LED(self.progressive_upright_led_pin)
        )
        falling_led = None if self.falling_led_pin is None else LED(self.falling_led_pin)
        cart_moving_right_led = None if self.cart_moving_right_led_pin is None else LED(self.cart_moving_right_led_pin)
        balance_led = None if self.balance_led_pin is None else LED(self.balance_led_pin)
        termination_led = None if self.termination_led_pin is None else LED(self.termination_led_pin)

        leds = [
            progressive_upright_led,
            falling_led,
            cart_moving_right_led,
            balance_led,
            termination_led
        ]

        return (
            RLock(),
            pca9685pw,
            DcMotor(
                driver=DcMotorDriverIndirectPCA9685PW(
                    pca9685pw=pca9685pw,
                    pwm_channel=self.motor_pwm_channel,
                    direction_pin=self.motor_pwm_direction_pin,
                    reverse=self.motor_negative_speed_is_right
                ),
                speed=0
            ),
            cart_rotary_encoder,
            pole_rotary_encoder,
            left_limit_switch,
            left_limit_pressed,
            left_limit_released,
            right_limit_switch,
            right_limit_pressed,
            right_limit_released,
            progressive_upright_led,
            falling_led,
            cart_moving_right_led,
            balance_led,
            termination_led,
            not self.load_calibration(),
            UltrasonicRangeFinder(
                trigger_pin=self.centering_range_finder_trigger_pin,
                echo_pin=self.centering_range_finder_echo_pin,
                measurements_per_second=4
            ),
            leds
        )

    def load_calibration(
            self
    ) -> bool:
        """
        Load calibration.

        :return: True if calibration was loaded.
        """

        loaded = False

        if os.path.exists(self.calibration_path):
            logging.info(f'Loading calibration values stored at {self.calibration_path}')
            try:
                with open(self.calibration_path, 'rb') as f:
                    self.__dict__.update(pickle.load(f))
                loaded = True
            except ValueError as e:
                logging.error(f'Error loading calibration values {e}')

        return loaded

    def get_state_space_dimensionality(
            self
    ) -> int:
        """
        Get the dimensionality of the state space.

        :return: Number of dimensions.
        """

        return 4

    def get_state_dimension_names(
            self
    ) -> List[str]:
        """
        Get names of state dimensions.

        :return: List of names.
        """

        return [
            'cart-position',
            'cart-velocity',
            'pole-angle',
            'pole-angular-velocity'
        ]

    def get_action_space_dimensionality(
            self
    ) -> int:
        """
        Get the dimensionality of the action space.

        :return: Number of dimensions.
        """

        return 1

    def get_action_dimension_names(
            self
    ) -> List[str]:
        """
        Get action names.

        :return: List of names.
        """

        return ['motor-speed-change']

    def calibrate(
            self
    ):
        """
        Calibrate the cart-pole apparatus, leaving the cart centered.
        """

        logging.info('Calibrating.')

        # create some space to do deadzone identification
        logging.info('Moving cart to right limit to create space for deadzone identification.')
        self.set_motor_speed(20)
        self.right_limit_pressed.wait()
        self.set_motor_speed(-20)
        time.sleep(3.0)
        self.set_motor_speed(0)
        logging.info('Deadzone space created.')

        # identify the minimum motor speeds that will get the cart to move left and right. there's a deadzone in the
        # middle that depends on the logic of the motor circuitry, mass and friction of the assembly, etc. this
        # nonlinearity of motor speed and cart velocity will confuse the controller. use the same speed in each
        # direction, choosing the max if they are different.
        deadzone_speed = max(
            abs(self.identify_motor_speed_deadzone_limit(CartDirection.LEFT)),
            abs(self.identify_motor_speed_deadzone_limit(CartDirection.RIGHT))
        )
        self.motor_deadzone_speed_left = -deadzone_speed
        self.motor_deadzone_speed_right = deadzone_speed

        # mark degrees at left and right limits. do this in whatever order is the most efficient given the cart's
        # current position. we can only do this efficiency trick after the initial calibration, since determining left
        # of center depends on having a value for the left limit. regardless, capture the rotary encoder's state at the
        # right and left limits for subsequent restoration.
        if self.left_limit_degrees is not None and CartPole.get_cart_position(
            self.cart_rotary_encoder.get_net_total_degrees(False),
            self.left_limit_degrees,
            self.cart_mm_per_degree,
            self.midline_mm
        ) == CartPosition.LEFT_OF_CENTER:
            self.move_cart_to_left_limit()
            self.left_limit_degrees = self.cart_rotary_encoder.get_net_total_degrees(False)
            self.cart_phase_change_index_at_left_limit = self.cart_rotary_encoder.phase_change_index.value
            self.move_cart_to_right_limit()
            self.right_limit_degrees = self.cart_rotary_encoder.get_net_total_degrees(False)
            self.cart_phase_change_index_at_right_limit = self.cart_rotary_encoder.phase_change_index.value
            cart_position = CartPosition.RIGHT_OF_CENTER
        else:
            self.move_cart_to_right_limit()
            self.right_limit_degrees = self.cart_rotary_encoder.get_net_total_degrees(False)
            self.cart_phase_change_index_at_right_limit = self.cart_rotary_encoder.phase_change_index.value
            self.move_cart_to_left_limit()
            self.left_limit_degrees = self.cart_rotary_encoder.get_net_total_degrees(False)
            self.cart_phase_change_index_at_left_limit = self.cart_rotary_encoder.phase_change_index.value
            cart_position = CartPosition.LEFT_OF_CENTER

        # calibrate mm/degree and the midline
        self.limit_to_limit_degrees = abs(self.left_limit_degrees - self.right_limit_degrees)
        self.cart_mm_per_degree = self.limit_to_limit_mm / self.limit_to_limit_degrees
        self.midline_degrees = (self.left_limit_degrees + self.right_limit_degrees) / 2.0

        # identify maximum cart speed
        logging.info('Identifying maximum cart speed...')
        self.set_motor_speed(
            speed=-100 if cart_position == CartPosition.RIGHT_OF_CENTER else 100,
            acceleration_interval=timedelta(seconds=0.5)
        )
        self.cart_rotary_encoder.wait_for_cart_to_cross_center(
            original_position=cart_position,
            left_limit_degrees=self.left_limit_degrees,
            cart_mm_per_degree=self.cart_mm_per_degree,
            midline_mm=self.midline_mm,
            check_delay_seconds=0.1
        )
        cart_state: MultiprocessRotaryEncoder.State = self.cart_rotary_encoder.state
        self.max_cart_speed_mm_per_second = abs(cart_state.angular_velocity * self.cart_mm_per_degree)
        self.stop_cart()

        # set central phase change indices
        self.cart_phase_change_index_at_center = int(
            self.midline_degrees * self.cart_rotary_encoder.phase_changes_per_degree
        )
        self.pole_phase_change_index_at_bottom = self.pole_rotary_encoder.speed_encoder.phase_change_index.value
        self.pole_degrees_at_bottom = self.pole_rotary_encoder.speed_encoder.get_degrees(False)

        calibration = {
            'motor_deadzone_speed_left': self.motor_deadzone_speed_left,
            'motor_deadzone_speed_right': self.motor_deadzone_speed_right,
            'left_limit_degrees': self.left_limit_degrees,
            'cart_phase_change_index_at_left_limit': self.cart_phase_change_index_at_left_limit,
            'right_limit_degrees': self.right_limit_degrees,
            'cart_phase_change_index_at_right_limit': self.cart_phase_change_index_at_right_limit,
            'limit_to_limit_degrees': self.limit_to_limit_degrees,
            'cart_mm_per_degree': self.cart_mm_per_degree,
            'midline_degrees': self.midline_degrees,
            'max_cart_speed_mm_per_second': self.max_cart_speed_mm_per_second,
            'cart_phase_change_index_at_center': self.cart_phase_change_index_at_center,
            'pole_phase_change_index_at_bottom': self.pole_phase_change_index_at_bottom,
            'pole_degrees_at_bottom': self.pole_degrees_at_bottom
        }

        logging.info(f'Calibration:  {calibration}')

        if self.calibration_path != '':
            logging.info(f'Saving calibration at {self.calibration_path}')
            try:
                with open(self.calibration_path, 'wb') as f:
                    pickle.dump(calibration, f)
            except ValueError as e:
                logging.error(f'Error saving calibration:  {e}')

    def identify_motor_speed_deadzone_limit(
            self,
            direction: CartDirection
    ) -> int:
        """
        Identify the deadzone in a direction.

        :param direction: Direction.
        :return: Motor speed that cause the cart to begin moving in the given direction.
        """

        self.stop_cart()

        if direction == CartDirection.LEFT:
            increment = -1
            limit_switch = self.left_limit_switch
        elif direction == CartDirection.RIGHT:
            increment = 1
            limit_switch = self.right_limit_switch
        else:
            raise ValueError(f'Unknown direction:  {direction}')

        moving_ticks_required = 5
        moving_ticks_remaining = moving_ticks_required
        speed = self.motor.get_speed()
        self.cart_rotary_encoder.update_state(True)
        while moving_ticks_remaining > 0:
            time.sleep(0.5)
            assert not limit_switch.is_pressed()
            if abs(self.cart_rotary_encoder.get_angular_velocity(True)) < 50.0:
                moving_ticks_remaining = moving_ticks_required
                speed += increment
                self.set_motor_speed(speed)
            else:
                moving_ticks_remaining -= 1

        self.stop_cart()

        logging.info(f'Deadzone ends at speed {speed} in the {direction} direction.')

        return speed

    def move_cart_to_left_limit(
            self
    ):
        """
        Move the cart to the left limit.
        """

        if not self.left_limit_pressed.is_set():
            logging.info('Moving cart to the left and waiting for limit switch.')
            self.set_motor_speed(2 * self.motor_deadzone_speed_left)
            self.left_limit_pressed.wait()

        logging.info('Moving cart away from left limit switch.')
        self.set_motor_speed(self.motor_deadzone_speed_right)
        self.left_limit_released.wait()
        self.stop_cart()

    def left_limit_event(
            self,
            is_pressed: bool
    ):
        """
        Receive an event from the left limit switch.

        :param is_pressed: Whether the limit switch is pressed.
        """

        self.handle_limit_event(
            'Left',
            is_pressed,
            self.left_limit_pressed,
            self.left_limit_released
        )

    def move_cart_to_right_limit(
            self
    ):
        """
        Move the cart to the right limit.
        """

        if not self.right_limit_pressed.is_set():
            logging.info('Moving cart to the right and waiting for limit switch.')
            self.set_motor_speed(2 * self.motor_deadzone_speed_right)
            self.right_limit_pressed.wait()

        logging.info('Moving cart away from right limit switch.')
        self.set_motor_speed(self.motor_deadzone_speed_left)
        self.right_limit_released.wait()
        self.stop_cart()

    def right_limit_event(
            self,
            is_pressed: bool
    ):
        """
        Receive an event from the right limit switch.

        :param is_pressed: Whether the limit switch is pressed.
        """

        self.handle_limit_event(
            'Right',
            is_pressed,
            self.right_limit_pressed,
            self.right_limit_released
        )

    def handle_limit_event(
            self,
            descriptor: str,
            is_pressed: bool,
            limit_pressed: Event,
            limit_released: Event
    ):
        """
        Handle a limit event.

        :param descriptor: Limit descriptor for logging.
        :param is_pressed: Whether the limit is pressed (True) or released (False).
        :param limit_pressed: Pressed event.
        :param limit_released: Released event.
        """

        if is_pressed:

            logging.info(f'{descriptor} limit pressed.')

            with self.state_lock:

                # it's important to stop the cart any time the limit switch is pressed
                self.stop_cart()

                # hitting a limit switch in the middle of an episode means that we've lost calibration. the soft limits
                # should have prevented this, but this failed. end the episode and calibrate upon the next episode
                # reset.
                self.state: MdpState
                if self.state is not None and not self.state.terminal:
                    self.state = self.get_state(t=None, terminal=True, update_velocity_and_acceleration=False)
                    self.calibrate_on_next_reset = True

            # another thread may attempt to wait for the switch to be released immediately upon the pressed event being
            # set. prevent a race condition by first clearing the released event before setting the pressed event.
            limit_released.clear()
            limit_pressed.set()

        else:

            logging.info(f'{descriptor} limit released.')

            # another thread may attempt to wait for the switch to be pressed immediately upon the released event being
            # set. prevent a race condition by first clearing the pressed event before setting the released event.
            limit_pressed.clear()
            limit_released.set()

    def center_cart(
            self,
            restore_limit_state: bool,
            restore_center_state: bool
    ):
        """
        Center the cart.

        :param restore_limit_state: Whether to restore the limit state before centering the cart. Doing this ensures
        that the centering calculation will be accurate and that any out-of-calibration issues will be mitigated. This
        is somewhat expensive, since it involves physically moving the cart to a nearby limit switch before moving the
        cart to the center.
        :param restore_center_state: Whether to restore the center state after centering. This ensures that the initial
        movements away from the center will be equivalent to previous such movements at that position.
        """

        assert self.left_limit_degrees is not None, 'Must calibrate before centering.'
        assert self.cart_mm_per_degree is not None, 'Must calibrate before centering.'

        original_position = CartPole.get_cart_position(
            self.cart_rotary_encoder.get_net_total_degrees(False),
            self.left_limit_degrees,
            self.cart_mm_per_degree,
            self.midline_mm
        )

        if original_position == CartPosition.CENTERED:
            logging.info('Cart already centered.')
            return

        logging.info('Centering cart.')

        # restore the limit state by physically positioning the cart at the limit and restoring the state to what it
        # was when we calibrated. this corrects any loss of calibration that occurred while moving the cart. do this in
        # whatever order is the most efficient given the cart's current position.
        if restore_limit_state:

            logging.info('Restoring limit state.')

            if original_position == CartPosition.LEFT_OF_CENTER:
                self.move_cart_to_left_limit()
                self.cart_rotary_encoder.phase_change_index.value = self.cart_phase_change_index_at_left_limit
            else:
                self.move_cart_to_right_limit()
                self.cart_rotary_encoder.phase_change_index.value = self.cart_phase_change_index_at_right_limit

            self.cart_rotary_encoder.update_state(False)

        while original_position != CartPosition.CENTERED:
            original_position = self.center_cart_at_speed(True, original_position)
            if original_position != CartPosition.CENTERED:
                logging.info('Failed to center cart. Trying again.')
        else:
            logging.info('Cart centered.')

        logging.info('Waiting for stationary pole.')
        self.pole_rotary_encoder.speed_encoder.wait_for_stationarity()

        if restore_center_state:
            self.cart_rotary_encoder.phase_change_index.value = self.cart_phase_change_index_at_center
            self.cart_rotary_encoder.update_state(False)
            self.pole_rotary_encoder.speed_encoder.phase_change_index.value = self.pole_phase_change_index_at_bottom
            self.pole_rotary_encoder.speed_encoder.update_state(False)

        logging.info(
            f'Pole is stationary at degrees:  '
            f'{self.pole_rotary_encoder.speed_encoder.get_net_total_degrees(False):.1f}'
        )

    def center_cart_at_speed(
            self,
            fast: bool,
            original_position: CartPosition
    ) -> CartPosition:
        """
        Center the cart.

        :param fast: Center cart quickly (True) but with lower accuracy or slowly (False) and with higher accuracy.
        :param original_position: Original cart position.
        :return: Final position.
        """

        assert self.left_limit_degrees is not None, 'Must calibrate before centering.'
        assert self.cart_mm_per_degree is not None, 'Must calibrate before centering.'

        if original_position == CartPosition.CENTERED:
            logging.info('Cart already centered.')
            return original_position

        if original_position == CartPosition.LEFT_OF_CENTER:
            centering_speed = self.motor_deadzone_speed_right
        else:
            centering_speed = self.motor_deadzone_speed_left

        if fast:
            centering_speed *= 3

        # move toward the center, wait for the center to be reached, and stop the cart.
        logging.info(f'Centering cart at speed:  {centering_speed}')
        self.set_motor_speed(centering_speed)
        while (
            (
                (distance := self.centering_range_finder.measure_distance_once()) is None or
                distance > 5.0
            ) and
            not self.left_limit_pressed.is_set() and
            not self.right_limit_pressed.is_set()
        ):
            if distance is not None:
                logging.debug(f'Centering distance:  {distance:.1f} cm')

            time.sleep(0.2)

        self.stop_cart()

        if distance is None:
            logging.info('No centering distance. Must have hit a limit switch.')
        else:
            logging.info(f'Centered with distance:  {distance:.1f} cm')

        if self.left_limit_pressed.is_set():
            self.move_cart_to_left_limit()
            centered_position = CartPosition.LEFT_OF_CENTER
        elif self.right_limit_pressed.is_set():
            self.move_cart_to_right_limit()
            centered_position = CartPosition.RIGHT_OF_CENTER
        else:
            centered_position = CartPosition.CENTERED

        return centered_position

    def stop_cart(
            self
    ):
        """
        Stop the cart and do not return until it is stationary.
        """

        logging.info('Stopping cart.')

        # we've observed cases in which setting the motor speed fails, or it does not fail but the subsequent wait
        # fails due to interprocess communication failure. in either case, the present process faults without stopping
        # the cart, and the cart physically slams into the rail's end and the motor continues driving it. this is a
        # critical failure. disable the motor pwm to ensure the motor controller is off.
        self.disable_motor_pwm()

        # noinspection PyBroadException
        try:
            self.set_motor_speed(0)
            self.cart_rotary_encoder.wait_for_stationarity()
        except Exception as e:

            logging.critical(f'Failed to stop cart. Leaving the motor PWM disabled. Exception {e}')

            # reraise the error. this will leave the pwm disabled forever, which is precisely what we want since we've
            # experienced a critical failure.
            raise e

        else:

            logging.info('Cart stopped.')

            # enable the failsafe pwm now that we've set the motor speed to zero and the cart has stopped
            self.enable_motor_pwm()

    def disable_motor_pwm(
            self
    ):
        """
        Disable the motor PWM.
        """

        gpio.output(self.failsafe_pwm_off_pin, gpio.HIGH)
        logging.info('Motor PWM disabled.')

    def enable_motor_pwm(
            self
    ):
        """
        Enable the motor PWM.
        """

        gpio.output(self.failsafe_pwm_off_pin, gpio.LOW)
        logging.info('Motor PWM enabled.')

    def set_motor_speed(
            self,
            speed: int,
            acceleration_interval: Optional[timedelta] = None
    ):
        """
        Set the motor speed.

        :param speed: Speed.
        :param acceleration_interval: Interval of time in which to change to the given speed, or None to do it
        instantaneously.
        """

        if speed < 0 and self.left_limit_switch.is_pressed():
            logging.info('Left limit is pressed. Cannot set motor speed negative.')
        elif speed > 0 and self.right_limit_switch.is_pressed():
            logging.info('Right limit is pressed. Cannot set motor speed positive.')
        else:

            if acceleration_interval is None:
                intermediate_speeds = [speed]
                per_speed_sleep_seconds = None
            else:
                intermediate_speeds = list(dict.fromkeys([
                    int(intermediate_speed)
                    for intermediate_speed in np.linspace(
                        start=self.motor.get_speed(),
                        stop=speed,
                        num=10,
                        endpoint=True
                    )
                ]))
                per_speed_sleep_seconds = acceleration_interval.total_seconds() / len(intermediate_speeds)

            for intermediate_speed in intermediate_speeds:

                # we occasionally get i/o errors from the underlying interface to the pwm. this probably has something
                # to do with attempting to write new values at a high rate like we're doing here. catch any such error
                # and try again.
                attempt = 0
                while attempt < 5:

                    try:
                        self.motor.set_speed(intermediate_speed)
                        break
                    except OSError as e:
                        logging.error(f'Error while setting speed:  {e}')
                        time.sleep(0.1)

                    attempt += 1

                else:
                    raise ValueError(f'Failed to set motor speed after {attempt} attempts.')

                if per_speed_sleep_seconds is not None:
                    time.sleep(per_speed_sleep_seconds)

            # the cart's rotary encoder uses uniphase tracking. it requires an external signal telling it which
            # direction it is moving. provide that signal based on the input speed of the motor. this assumes negligible
            # latency between setting the motor pwm input and registering the effect on rotary encoder output.
            self.cart_rotary_encoder.clockwise.value = speed > 0

    def turn_off_leds(
            self
    ):
        """
        Turn off the LEDs.
        """

        for led in self.leds:
            CartPole.set_led(led, False)

    def flash_leds(
            self
    ):
        """
        Flash LEDs.
        """

        self.turn_off_leds()

        for _ in range(5):
            for led in self.leds:
                CartPole.set_led(led, True)
                time.sleep(0.1)
                CartPole.set_led(led, False)

    @staticmethod
    def set_led(
            led: Optional[LED],
            on: bool
    ):
        """
        Set LED on/off.

        :param led: LED (can be None).
        :param on: Whether to turn on (True) or off (False).
        """

        if led is not None:
            if on:
                led.turn_on()
            else:
                led.turn_off()

    def reset_for_new_run(
            self,
            agent: Any
    ) -> MdpState:
        """
        Reset the environment to its initial conditions.

        :param agent: Agent used to generate on-the-fly state identifiers.
        :return: Initial state.
        """

        logging.info(f'Reset {self.num_resets + 1}.')

        super().reset_for_new_run(self.agent)
        self.agent = agent

        self.flash_leds()

        self.plot_label_data_kwargs['Motor Speed'] = (
            dict(),
            {
                'linewidth': 0.5
            }
        )

        self.plot_label_data_kwargs['Pole Angle * 10'] = (
            dict(),
            {
                'linewidth': 0.5
            }
        )

        self.plot_label_data_kwargs['Pole Angular Vel.'] = (
            dict(),
            {
                'linewidth': 0.5
            }
        )

        self.plot_label_data_kwargs['Pole Angular Acc.'] = (
            dict(),
            {
                'linewidth': 0.5
            }
        )

        self.fraction_time_balancing.reset()

        if self.original_agent_gamma is None:
            self.original_agent_gamma = self.agent.gamma

        # reset original agent gamma value. we manipulate gamma during episode phases and for post-truncation
        # convergence.
        self.agent.gamma = self.original_agent_gamma
        logging.info(f'Restored agent.gamma to {self.agent.gamma}.')

        # if the previous episode achieved progressive upright, then reduce the angle down to the balance angle.
        if self.achieved_progressive_upright:
            self.achieved_progressive_upright = False
            if self.progressive_upright_pole_angle == self.balance_pole_angle:
                logging.info(
                    f'Progressive upright pole angle is already {self.progressive_upright_pole_angle}. Not reducing.'
                )
            else:
                self.progressive_upright_pole_angle = max(
                    self.balance_pole_angle,
                    self.progressive_upright_pole_angle - 1.0
                )
                logging.info(
                    f'Reduced progressive upright pole angle to {self.progressive_upright_pole_angle} degrees.'
                )

        self.episode_phase = EpisodePhase.SWING_UP
        self.lost_balance_timestamp = None

        self.motor.start()

        # calibrate if needed
        if self.calibrate_on_next_reset:
            self.calibrate()
            self.calibrate_on_next_reset = False

        # center the cart with the current calibration and reset the rotary encoders to their calibration-initial
        # conditions at center. don't bother to restore the limit state, as this significantly slows down the centering.
        # just defer any errors until we hit a hard side limit, at which time we'll recalibrate.
        self.center_cart(False, True)

        self.state = self.get_state(t=None, terminal=False, update_velocity_and_acceleration=False)
        self.previous_timestep_epoch = None
        self.current_timesteps_per_second.reset()

        logging.info(f'State after reset:  {self.state}')

        return self.state

    def cart_violates_soft_limit(
            self,
            cart_mm_from_center: float
    ) -> bool:
        """
        Get whether a distance from center is terminal.

        :param cart_mm_from_center: Distance (mm) from center.
        """

        return abs(cart_mm_from_center) >= self.soft_limit_mm_from_midline

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

        with self.state_lock:

            previous_state = self.state

            # update the current state if we haven't yet terminated
            if not previous_state.terminal:
                self.state = self.get_state(t=t, terminal=None, update_velocity_and_acceleration=True)
                CartPole.set_led(self.falling_led, self.state.pole_is_falling)
                CartPole.set_led(self.cart_moving_right_led, self.state.cart_velocity_mm_per_second > 0.0)

            new_termination = not previous_state.terminal and self.state.terminal
            new_truncation = not previous_state.truncated and self.state.truncated

            if new_termination:
                CartPole.set_led(self.termination_led, True)
                self.stop_cart()

            if new_truncation:

                logging.info('Truncated.')
                self.time_step_axv_lines[t] = {
                    'color': 'yellow',
                    'label': 'Truncated'
                }

                # post-truncation convergence to zero takes too long with gammas close to 1.0 and a slow physical
                # system. allow decreased gamma to obtain faster convergence to zero.
                if self.truncation_gamma is not None and self.truncation_gamma != self.agent.gamma:
                    self.agent.gamma = self.truncation_gamma
                    logging.info(f'Set agent.gamma to {self.agent.gamma} to obtain faster convergence to zero.')

            # perform nominal environment advancement if we haven't terminated. we continue to do this after truncation,
            # since we're waiting for the learning procedure to exit the episode.
            if not self.state.terminal:

                # extract the desired speed change from the action
                assert isinstance(a, ContinuousMultiDimensionalAction)
                assert a.value.shape == (1,)
                desired_speed_change = round(float(a.value[0]))

                # constrain the speed change to the range of the motor
                current_speed = self.motor.get_speed()
                desired_speed = current_speed + desired_speed_change
                constrained_speed = self.motor.constrain_speed(desired_speed)
                if constrained_speed == desired_speed:
                    speed_change = desired_speed_change
                    next_speed = desired_speed
                else:
                    speed_change = constrained_speed - current_speed
                    next_speed = constrained_speed

                    # if we constrained the speed, then reflect this in the agent's action so that the learning
                    # procedure correctly interprets the environment's feedback. this is like a feedback signal from the
                    # environment that the desired action was not possible.
                    a.value = np.array([speed_change])

                # if the next speed falls into the motor's deadzone, bump it to the minimum speed based on the direction
                # of speed change. this ensures linearity between the agent's action and the cart's response. we
                # shouldn't reflect this modification in the agent's action value. from the agent's perspective, all
                # actions (speed changes) are relative to the current speed, and the deadzone does not exist.
                if self.motor_deadzone_speed_left < next_speed < self.motor_deadzone_speed_right:
                    if speed_change < 0:
                        next_speed = self.motor_deadzone_speed_left
                    elif speed_change > 0:
                        next_speed = self.motor_deadzone_speed_right

                self.set_motor_speed(next_speed)

            # adapt the sleep time to obtain the desired steps per second
            if self.previous_timestep_epoch is None:
                self.previous_timestep_epoch = time.time()
            else:

                # update current timesteps per second
                current_timestep_epoch = time.time()
                self.current_timesteps_per_second.update(1.0 / (current_timestep_epoch - self.previous_timestep_epoch))
                self.previous_timestep_epoch = current_timestep_epoch

                # adapt the timestep sleep duration to achieve the target steps per second, given the overhead involved
                # in executing each call to advance.
                if self.current_timesteps_per_second > self.timesteps_per_second:
                    self.timestep_sleep_seconds *= 1.01
                elif self.current_timesteps_per_second < self.timesteps_per_second:
                    self.timestep_sleep_seconds *= 0.99

                logging.debug(f'Running at {self.current_timesteps_per_second:.1f} steps/sec')

            time.sleep(self.timestep_sleep_seconds)

            # calculate reward
            reward_value = self.get_reward(self.state)

            logging.debug(f'State {t}:  {self.state}')
            logging.debug(f'Reward {t}:  {reward_value}')

            self.plot_label_data_kwargs['Motor Speed'][0][t] = self.motor.get_speed()
            self.plot_label_data_kwargs['Pole Angle * 10'][0][t] = self.state.pole_angle_deg_from_upright * 10.0
            self.plot_label_data_kwargs['Pole Angular Vel.'][0][t] = self.state.pole_angular_velocity_deg_per_sec
            self.plot_label_data_kwargs['Pole Angular Acc.'][0][t] = (
                self.state.pole_angular_acceleration_deg_per_sec_squared
            )

            self.fraction_time_balancing.update(float(self.episode_phase == EpisodePhase.BALANCE))
            if self.state.terminal:
                self.metric_value['Fraction Balancing'] = self.fraction_time_balancing.get_value()

            return self.state, Reward(None, reward_value)

    @staticmethod
    def get_reward(
            state: CartPoleState
    ) -> float:
        """
        Get reward for a state.

        :param state: State.
        :return: Reward.
        """

        if state.terminal:
            reward = -1.0
        else:
            reward = (
                state.zero_to_one_pole_angle *
                state.zero_to_one_pole_angular_speed
            )

        return reward

    def exiting_episode_without_termination(
            self
    ):
        """
        Called when a learning procedure is exiting the episode without natural termination (e.g., after truncation).
        The episode will not reach a natural termination state. Instead, the episode loop will exit. This function is
        called to provide the environment an opportunity to clean up resources. This is not usually needed with
        simulation-based environments since breaking the episode loop prevents any further episode advancement. However,
        in physical environments the system might continue to advance in the absence of further calls to the advance
        function. This function allows the environment to perform any adjustments that are normally required upon
        termination.
        """

        self.stop_cart()

    def get_state(
            self,
            t: Optional[int],
            terminal: Optional[bool],
            update_velocity_and_acceleration: bool
    ) -> CartPoleState:
        """
        Get the current state.

        :param t: Time step to consider for episode truncation, or None if not in an episode.
        :param terminal: Whether to force a terminal state, or None for natural assessment.
        :param update_velocity_and_acceleration: Whether to update velocity and acceleration estimates.
        :return: State.
        """

        assert self.agent is not None

        self.cart_rotary_encoder.update_state(update_velocity_and_acceleration)
        cart_state: MultiprocessRotaryEncoder.State = self.cart_rotary_encoder.state

        self.pole_rotary_encoder.update_state(update_velocity_and_acceleration)
        pole_state: MultiprocessRotaryEncoder.State = self.pole_rotary_encoder.state

        cart_mm_from_left_limit = abs(
            cart_state.net_total_degrees - self.left_limit_degrees
        ) * self.cart_mm_per_degree

        cart_mm_from_center = cart_mm_from_left_limit - self.limit_to_limit_mm / 2.0

        # get pole's degree from vertical at bottom
        pole_angle_deg_from_bottom = pole_state.degrees - self.pole_degrees_at_bottom

        # translate to [-180,180] degrees from bottom. if the degrees value is beyond these bounds, then subtract a
        # complete rotation in the appropriate direction in order to get the same degree orientation but within the
        # desired bounds.
        if abs(pole_angle_deg_from_bottom) > 180.0:
            pole_angle_deg_from_bottom -= np.sign(pole_angle_deg_from_bottom) * 360.0

        # convert to degrees from upright, which is what we need for the reward calculation.
        if pole_angle_deg_from_bottom == 0.0:
            pole_angle_deg_from_upright = 180.0  # equivalent to -180.0
        else:
            pole_angle_deg_from_upright = -np.sign(pole_angle_deg_from_bottom) * 180.0 + pole_angle_deg_from_bottom

        pole_is_progressive_upright = abs(pole_angle_deg_from_upright) <= self.progressive_upright_pole_angle
        pole_is_balancing = abs(pole_angle_deg_from_upright) <= self.balance_pole_angle
        pole_is_balancing_slowly = abs(pole_state.angular_velocity) <= self.balance_angular_velocity

        # swing up:  we're in the swing-up phase if we're not progressive upright (which implies also not balancing). we
        # start each episode in the swing-up phase, so this will only apply if we've fallen below upright.
        if self.episode_phase != EpisodePhase.SWING_UP and not pole_is_progressive_upright:

            self.episode_phase = EpisodePhase.SWING_UP

            # if this is the first time we went back to swing-up, then start the lost-balance timer.
            if self.lost_balance_timestamp is None:
                self.lost_balance_timestamp = time.time()
                self.time_step_axv_lines[t] = {
                    'color': 'purple',
                    'linestyle': '--',
                    'label': 'Lost upright'
                }
                logging.info(
                    f'Pole has lost its upright position. Angle {pole_angle_deg_from_upright:.2f} exceeds the maximum '
                    f'allowable of {self.progressive_upright_pole_angle:.1f}. Starting lost-balance timer of '
                    f'{self.lost_balance_timer_seconds} seconds.'
                )

            for led in [self.progressive_upright_led, self.balance_led]:
                CartPole.set_led(led, False)

        # progressive upright:  anything upright, including balancing but moving too quickly. the balance policy is only
        # for balancing slowly.
        if (
            self.episode_phase != EpisodePhase.PROGRESSIVE_UPRIGHT and
            (
                (pole_is_progressive_upright and not pole_is_balancing) or
                (pole_is_balancing and not pole_is_balancing_slowly)
            )
        ):
            self.episode_phase = EpisodePhase.PROGRESSIVE_UPRIGHT
            self.achieved_progressive_upright = True
            logging.info(f'Progressive upright @ {pole_angle_deg_from_upright:.1f} degrees.')
            self.time_step_axv_lines[t] = {
                'color': 'purple',
                'label': 'Progressive upright'
            }
            CartPole.set_led(self.progressive_upright_led, True)
            CartPole.set_led(self.balance_led, False)

        # balancing:  above the balance threshold and moving slowly.
        if self.episode_phase != EpisodePhase.BALANCE and pole_is_balancing and pole_is_balancing_slowly:
            self.episode_phase = EpisodePhase.BALANCE
            logging.info(
                f'Balancing @ {pole_angle_deg_from_upright:.1f} deg @ {pole_state.angular_velocity:.1f} deg/sec.'
            )
            self.time_step_axv_lines[t] = {
                'color': 'blue',
                'label': 'Balance'
            }
            CartPole.set_led(self.balance_led, True)
            CartPole.set_led(self.progressive_upright_led, False)
            if self.balance_gamma != self.agent.gamma:
                self.agent.gamma = self.balance_gamma
                logging.info(f'Set agent.gamma={self.agent.gamma}.')

        truncated = False

        # truncate due to timer
        if (
            self.lost_balance_timestamp is not None and
            (time.time() - self.lost_balance_timestamp) >= self.lost_balance_timer_seconds
        ):
            truncated = True

        # truncate due to time steps
        if t is not None and self.T is not None and t >= self.T:
            truncated = True

        # terminate at violation of soft limit, since continuing might cause the cart to physically impact the
        # limit switch at a high speed. only do this if a termination value isn't being forced by the caller.
        if terminal is None:
            terminal = self.cart_violates_soft_limit(cart_mm_from_center)
            if terminal:
                logging.info(
                    f'Cart distance from center ({abs(cart_mm_from_center):.1f} mm) exceeds soft limit '
                    f'({self.soft_limit_mm_from_midline} mm). Terminating.'
                )

        return CartPoleState(
            environment=self,
            cart_mm_from_center=cart_mm_from_center,
            cart_velocity_mm_per_sec=cart_state.angular_velocity * self.cart_mm_per_degree,
            pole_angle_deg_from_upright=pole_angle_deg_from_upright,
            pole_angular_velocity_deg_per_sec=pole_state.angular_velocity,
            pole_angular_acceleration_deg_per_sec_squared=pole_state.angular_acceleration,
            step=t,
            agent=self.agent,
            terminal=terminal,
            truncated=truncated
        )

    def close(
            self
    ):
        """
        Close the environment, releasing resources.
        """

        # noinspection PyBroadException
        try:
            self.cart_rotary_encoder.wait_for_termination()
        except Exception:
            pass

        # noinspection PyBroadException
        try:
            self.pole_rotary_encoder.wait_for_termination()
        except Exception:
            pass

        cleanup()
