import logging
import time
from argparse import ArgumentParser
from threading import Event, RLock
from typing import List, Tuple, Any, Optional, Dict

import numpy as np
from numpy.random import RandomState
from smbus2 import SMBus

from raspberry_py.gpio import CkPin
from raspberry_py.gpio import Event as RpyEvent
from raspberry_py.gpio.controls import LimitSwitch
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

    def __str__(
            self
    ) -> str:
        """
        Get string.

        :return: String.
        """

        return (
            f'Cart x={self.observation[0]:.1f} mm @ {self.observation[1]:.1f} mm/s; '
            f'Pole deg={self.observation[2]:.1f} @ {self.observation[2]:.1f} deg/s'
        )


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

        self.midline_mm = self.limit_to_limit_mm / 2.0
        self.soft_limit_mm_from_midline = self.midline_mm - 100.0
        self.move_to_limit_motor_speed = 25
        self.move_away_from_limit_motor_speed = 15
        self.agent: Optional[MdpAgent] = None
        self.state_lock = RLock()

        self.actions = [
            ContinuousMultiDimensionalAction(
                value=None,
                min_values=np.array([-5.0]),
                max_values=np.array([5.0]),
                name='speed-change'
            )
        ]

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

        self.cart_rotary_encoder = RotaryEncoder(
            phase_a_pin=self.cart_rotary_encoder_phase_a_pin,
            phase_b_pin=self.cart_rotary_encoder_phase_b_pin,
            phase_changes_per_rotation=2400,
            report_state=lambda e: False,
            degrees_per_second_smoothing=0.75
        )
        self.cart_rotary_encoder_centered_state: Optional[Dict[str, float]] = None

        self.pole_rotary_encoder = RotaryEncoder(
            phase_a_pin=self.pole_rotary_encoder_phase_a_pin,
            phase_b_pin=self.pole_rotary_encoder_phase_b_pin,
            phase_changes_per_rotation=2400,
            report_state=lambda e: False,
            degrees_per_second_smoothing=0.75
        )
        self.pole_rotary_encoder_stationary_state: Optional[Dict[str, float]] = None
        self.pole_rotary_encoder_degrees_at_bottom: Optional[float] = None

        self.left_limit_switch = LimitSwitch(
            input_pin=self.left_limit_switch_input_pin,
            bounce_time_ms=5
        )
        self.left_limit_pressed = Event()
        self.left_limit_released = Event()
        if self.left_limit_switch.is_pressed():
            self.left_limit_pressed.set()
        else:
            self.left_limit_released.set()
        self.left_limit_switch.event(lambda s: self.left_limit_event(s.pressed))

        self.right_limit_switch = LimitSwitch(
            input_pin=self.right_limit_switch_input_pin,
            bounce_time_ms=5
        )
        self.right_limit_pressed = Event()
        self.right_limit_released = Event()
        if self.right_limit_switch.is_pressed():
            self.right_limit_pressed.set()
        else:
            self.right_limit_released.set()
        self.right_limit_switch.event(lambda s: self.right_limit_event(s.pressed))

        self.center_reached = Event()

        # calibration state and values
        self.has_calibrated = False
        self.calibrate_on_next_reset = True
        self.left_limit_degrees: Optional[float] = None
        self.right_limit_degrees: Optional[float] = None
        self.limit_to_limit_degrees: Optional[float] = None
        self.cart_mm_per_degree: Optional[float] = None
        self.midline_degrees: Optional[float] = None

    def calibrate(
            self
    ):
        """
        Calibrate the cart-pole apparatus.
        """

        logging.info('Calibrating.')

        # mark degrees at left and right limits
        self.move_cart_to_left_limit()
        self.left_limit_degrees = self.cart_rotary_encoder.net_total_degrees
        self.move_cart_to_right_limit()
        self.right_limit_degrees = self.cart_rotary_encoder.net_total_degrees

        # calibrate mm/degree and the midline
        self.limit_to_limit_degrees = abs(self.left_limit_degrees - self.right_limit_degrees)
        self.cart_mm_per_degree = self.limit_to_limit_mm / self.limit_to_limit_degrees
        self.midline_degrees = (self.left_limit_degrees + self.right_limit_degrees) / 2.0

        # center cart and capture initial conditions of the rotary encoders for resetting later
        self.center_cart()
        self.cart_rotary_encoder_centered_state = self.cart_rotary_encoder.capture_state()
        self.pole_rotary_encoder_stationary_state = self.pole_rotary_encoder.capture_state()
        self.pole_rotary_encoder_degrees_at_bottom = self.pole_rotary_encoder.degrees

        logging.info(
            f'Calibrated:\n'
            f'\tLeft-limit degrees:  {self.left_limit_degrees}\n'
            f'\tRight-limit degrees:  {self.right_limit_degrees}\n'
            f'\tLimit-to-limit degrees:  {self.limit_to_limit_degrees}\n'
            f'\tCart mm / degree:  {self.cart_mm_per_degree}\n'
            f'\tMidline degree:  {self.midline_degrees}\n'
            f'\tPole degrees at bottom:  {self.pole_rotary_encoder_degrees_at_bottom}\n'
        )

        self.has_calibrated = True

    def move_cart_to_left_limit(
            self
    ):
        """
        Move the cart to the left limit.
        """

        if not self.left_limit_pressed.is_set():
            logging.info('Moving cart to the left and waiting for limit switch.')
            self.motor.set_speed(-self.move_to_limit_motor_speed)
            self.left_limit_pressed.wait()

        logging.info('Moving cart away from left limit switch.')
        self.motor.set_speed(self.move_away_from_limit_motor_speed)
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

            logging.info('Left limit pressed.')

            # any time the limit switch is pressed, it's important to stop the cart and terimate any current episode.
            # we can only do this after we have calibrated, since getting the state requires calibration values. if we
            # haven't calibrated, then we must have hit the limit switch from the calibration movements. those are
            # synchronized with the current event, and they will immediately move the cart away from the limit switch.
            # so there's no need to zero the speed here in that case.
            if self.has_calibrated:
                with self.state_lock:
                    self.motor.set_speed(0)
                    self.state = self.get_state(True)
                    self.calibrate_on_next_reset = True

            self.left_limit_released.clear()
            self.left_limit_pressed.set()

        else:

            logging.info('Left limit released.')

            self.left_limit_pressed.clear()
            self.left_limit_released.set()

    def move_cart_to_right_limit(
            self
    ):
        """
        Move the cart to the right limit.
        """

        if not self.right_limit_pressed.is_set():
            logging.info('Moving cart to the right and waiting for limit switch.')
            self.motor.set_speed(self.move_to_limit_motor_speed)
            self.right_limit_pressed.wait()

        logging.info('Moving cart away from right limit switch.')
        self.motor.set_speed(-self.move_away_from_limit_motor_speed)
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

            logging.info('Right limit pressed.')

            # any time the limit switch is pressed, it's important to stop the cart and terimate any current episode.
            # we can only do this after we have calibrated, since getting the state requires calibration values. if we
            # haven't calibrated, then we must have hit the limit switch from the calibration movements. those are
            # synchronized with the current event, and they will immediately move the cart away from the limit switch.
            # so there's no need to zero the speed here in that case.
            if self.has_calibrated:
                with self.state_lock:
                    self.motor.set_speed(0)
                    self.state = self.get_state(True)
                    self.calibrate_on_next_reset = True

            self.right_limit_released.clear()
            self.right_limit_pressed.set()

        else:

            logging.info('Right limit released.')

            self.right_limit_pressed.clear()
            self.right_limit_released.set()

    def center_cart(
            self
    ):
        """
        Center the cart.
        """

        logging.info('Centering cart.')

        # only report state when we cross the midline
        originally_left_of_center = self.is_left_of_center()
        self.cart_rotary_encoder.report_state = lambda _: (
            (
                originally_left_of_center and not self.is_left_of_center()
            )
            or
            (
                not originally_left_of_center and self.is_left_of_center()
            )
        )

        # move the cart until the state is reported (center is reached)
        self.center_reached.clear()
        centering_rpy_event = RpyEvent(lambda _: self.center_reached.set())
        self.cart_rotary_encoder.events.append(centering_rpy_event)
        self.motor.set_speed(
            self.move_to_limit_motor_speed if originally_left_of_center
            else -self.move_to_limit_motor_speed
        )
        self.center_reached.wait()

        # stop
        self.motor.set_speed(0)
        self.cart_rotary_encoder.report_state = lambda e: False
        self.cart_rotary_encoder.events.remove(centering_rpy_event)
        self.center_reached.clear()
        logging.info('Cart centered.\n')

        # wait for the pole to stop swinging
        logging.info('Waiting for stationary pole.')
        previous_pole_num_phase_changes = self.pole_rotary_encoder.num_phase_changes
        time.sleep(1.0)
        while self.pole_rotary_encoder.num_phase_changes != previous_pole_num_phase_changes:
            previous_pole_num_phase_changes = self.pole_rotary_encoder.num_phase_changes
            logging.info('Waiting for stationary pole.')
            time.sleep(1.0)
        logging.info(f'Pole is stationary at degrees:  {self.pole_rotary_encoder.net_total_degrees}\n')

    def is_left_of_center(
            self,
    ) -> bool:
        """
        Check whether the cart is left of the midline.

        :return: True if the cart is left of the midline and False otherwise.
        """

        return abs(
            self.cart_rotary_encoder.net_total_degrees - self.left_limit_degrees
        ) * self.cart_mm_per_degree < self.midline_mm

    def reset_for_new_run(
            self,
            agent: Any
    ) -> MdpState:
        """
        Reset the environment to a random nonterminal state, if any are specified, or to None.

        :param agent: Agent used to generate on-the-fly state identifiers.
        :return: Initial state.
        """

        logging.info(f'Reset {self.num_resets + 1}.')

        super().reset_for_new_run(self.agent)

        self.motor.start()

        if self.calibrate_on_next_reset:
            self.calibrate()
            self.calibrate_on_next_reset = False
        else:
            self.center_cart()
            self.cart_rotary_encoder.reset_state(self.cart_rotary_encoder_centered_state)
            self.pole_rotary_encoder.reset_state(self.pole_rotary_encoder_stationary_state)

        self.agent = agent
        self.state = self.get_state(False)

        logging.info(f'State after reset:  {self.state}')

        return self.state

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

            if self.state.terminal:

                reward_value = 0.0

            else:

                assert isinstance(a, ContinuousMultiDimensionalAction)
                assert a.value.shape == (1,)
                speed_change = round(float(a.value[0]))
                self.motor.set_speed(self.motor.get_speed() + speed_change)

                self.state = self.get_state(None)
                if self.state.terminal:
                    self.motor.set_speed(0)

                reward_value = 1.0

            return self.state, Reward(None, reward_value)

    def get_state(
            self,
            terminal: Optional[bool]
    ) -> CartPoleState:
        """
        Get the current state.

        :param terminal: Whether state is terminal, or None for natural assessment.
        :return: State.
        """

        mm_from_midline = (
            abs(self.cart_rotary_encoder.net_total_degrees - self.left_limit_degrees) * self.cart_mm_per_degree
        ) - self.limit_to_limit_mm / 2.0

        if terminal is None:
            terminal = abs(mm_from_midline) > self.soft_limit_mm_from_midline
            if terminal:
                logging.info('Cart exceeded soft limit. Terminating.')

        return CartPoleState(
            environment=self,
            observation=np.array([
                mm_from_midline,
                self.cart_rotary_encoder.degrees_per_second * self.cart_mm_per_degree,
                self.pole_rotary_encoder.degrees - self.pole_rotary_encoder_degrees_at_bottom,
                self.pole_rotary_encoder.degrees_per_second
            ]),
            agent=self.agent,
            terminal=terminal,
            truncated=False
        )
