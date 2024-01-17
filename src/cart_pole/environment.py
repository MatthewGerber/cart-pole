import logging
from argparse import ArgumentParser
from enum import Enum, auto
from multiprocessing import Process, Value, Pipe
# noinspection PyProtectedMember
from multiprocessing.connection import Connection
from threading import Event, RLock
import time
from typing import List, Tuple, Any, Optional, Dict, Callable

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


class MultiprocessRotaryEncoder(RotaryEncoder):

    def __init__(
            self,
            phase_a_pin: CkPin,
            phase_b_pin: CkPin,
            phase_changes_per_rotation: int,
            report_state: Optional[Callable[['RotaryEncoder'], bool]],
            degrees_per_second_smoothing: Optional[float],
            bounce_time_ms: Optional[float],
            net_total_degrees_value: Value,
            degrees_value: Value,
            degrees_per_second_value: Value
    ):
        super().__init__(
            phase_a_pin,
            phase_b_pin,
            phase_changes_per_rotation,
            report_state,
            degrees_per_second_smoothing,
            bounce_time_ms
        )

        self.net_total_degrees_value = net_total_degrees_value
        self.degrees_value = degrees_value
        self.degrees_per_second_value = degrees_per_second_value

        self.net_total_degrees_value.value = self.net_total_degrees
        self.degrees_value.value = self.degrees
        self.degrees_per_second_value.value = self.degrees_per_second

    def update_state(
            self
    ):
        super().update_state()

        self.net_total_degrees_value.value = self.net_total_degrees
        self.degrees_value.value = self.degrees
        self.degrees_per_second_value.value = self.degrees_per_second


class MultiprocessRotaryEncoderAPI:

    class CommandFunction(Enum):

        WAIT_FOR_STARTUP = auto()
        CAPTURE_STATE = auto()
        RESTORE_CAPTURED_STATE = auto()
        WAIT_FOR_STATIONARITY = auto()
        WAIT_FOR_CART_TO_CROSS_CENTER = auto()
        TERMINATE = auto()

    class Command:

        def __init__(
                self,
                function: 'MultiprocessRotaryEncoderAPI.CommandFunction',
                args: Optional[List[Any]] = None
        ):
            if args is None:
                args = []

            self.function = function
            self.args = args

    @staticmethod
    def loop(
            phase_a_pin: CkPin,
            phase_b_pin: CkPin,
            net_total_degrees_value: Value,
            degrees_value: Value,
            degrees_per_second_value: Value,
            command_pipe: Connection
    ):
        rotary_encoder = MultiprocessRotaryEncoder(
            phase_a_pin=phase_a_pin,
            phase_b_pin=phase_b_pin,
            phase_changes_per_rotation=2400,
            report_state=lambda e: False,
            degrees_per_second_smoothing=0.75,
            bounce_time_ms=None,
            net_total_degrees_value=net_total_degrees_value,
            degrees_value=degrees_value,
            degrees_per_second_value=degrees_per_second_value
        )

        while True:

            print('Waiting for command')

            command: MultiprocessRotaryEncoderAPI.Command = command_pipe.recv()

            if command.function == MultiprocessRotaryEncoderAPI.CommandFunction.WAIT_FOR_STARTUP:
                return_value = None
            elif command.function == MultiprocessRotaryEncoderAPI.CommandFunction.CAPTURE_STATE:
                return_value = rotary_encoder.capture_state()
            elif command.function == MultiprocessRotaryEncoderAPI.CommandFunction.RESTORE_CAPTURED_STATE:
                rotary_encoder.restore_captured_state(*command.args)
                return_value = None
            elif command.function == MultiprocessRotaryEncoderAPI.CommandFunction.WAIT_FOR_STATIONARITY:
                rotary_encoder.wait_for_stationarity(0.1)
                return_value = None
            elif command.function == MultiprocessRotaryEncoderAPI.CommandFunction.WAIT_FOR_CART_TO_CROSS_CENTER:

                (
                    originally_left_of_center,
                    left_limit_degrees,
                    cart_mm_per_degree,
                    midline_mm
                ) = command.args

                def is_left_of_center() -> bool:
                    """
                    Check whether the cart is left of the midline.

                    :return: True if the cart is left of the midline and False otherwise.
                    """

                    return abs(
                        rotary_encoder.net_total_degrees - left_limit_degrees
                    ) * cart_mm_per_degree < midline_mm

                # report state when we cross the midline
                rotary_encoder.report_state = lambda _: (
                    (
                        originally_left_of_center and not is_left_of_center()
                    )
                    or
                    (
                        not originally_left_of_center and is_left_of_center()
                    )
                )

                # wait for any state event to be reported and set the event
                center_reached = Event()
                center_reached_rpy_event = RpyEvent(lambda _: center_reached.set())
                rotary_encoder.events.append(center_reached_rpy_event)

                # wait for event
                center_reached.wait()

                # disable event
                rotary_encoder.report_state = lambda e: False
                rotary_encoder.events.remove(center_reached_rpy_event)

                return_value = None

            elif command.function == MultiprocessRotaryEncoderAPI.CommandFunction.TERMINATE:
                break
            else:
                raise ValueError(f'Unknown function:  {command.function}')

            command_pipe.send(return_value)

        command_pipe.send(None)

    def __init__(
            self,
            phase_a_pin: CkPin,
            phase_b_pin: CkPin
    ):
        self.phase_a_pin = phase_a_pin
        self.phase_b_pin = phase_b_pin

        self.net_total_degrees_value = Value('d', 0.0)
        self.degrees_value = Value('d', 0.0)
        self.degrees_per_second_value = Value('d', 0.0)

        self.parent_connection, self.child_connection = Pipe()

        self.process = Process(
            target=MultiprocessRotaryEncoderAPI.loop,
            args=(
                self.phase_a_pin,
                self.phase_b_pin,
                self.net_total_degrees_value,
                self.degrees_value,
                self.degrees_per_second_value,
                self.child_connection
            )
        )
        self.process.start()

    def get_net_total_degrees(
            self
    ) -> float:

        return self.net_total_degrees_value.value

    def get_degrees(
            self
    ) -> float:

        return self.degrees_value.value

    def get_degrees_per_second(
            self
    ) -> float:

        return self.degrees_per_second_value.value

    def wait_for_startup(
            self
    ):

        self.parent_connection.send(
            MultiprocessRotaryEncoderAPI.Command(MultiprocessRotaryEncoderAPI.CommandFunction.WAIT_FOR_STARTUP)
        )

        return_value = self.parent_connection.recv()

        assert return_value is None

    def capture_state(
            self
    ) -> Dict[str, float]:

        self.parent_connection.send(
            MultiprocessRotaryEncoderAPI.Command(MultiprocessRotaryEncoderAPI.CommandFunction.CAPTURE_STATE)
        )

        return self.parent_connection.recv()

    def restore_captured_state(
            self,
            captured_state: Dict[str, float]
    ):

        self.parent_connection.send(
            MultiprocessRotaryEncoderAPI.Command(
                MultiprocessRotaryEncoderAPI.CommandFunction.RESTORE_CAPTURED_STATE,
                [captured_state]
            )
        )

        return_value = self.parent_connection.recv()

        assert return_value is None

    def wait_for_stationarity(
            self
    ):

        self.parent_connection.send(
            MultiprocessRotaryEncoderAPI.Command(MultiprocessRotaryEncoderAPI.CommandFunction.WAIT_FOR_STATIONARITY)
        )

        return_value = self.parent_connection.recv()

        assert return_value is None

    def wait_for_cart_to_cross_center(
            self,
            originally_left_of_center: bool,
            left_limit_degrees: float,
            cart_mm_per_degree: float,
            midline_mm: float
    ):

        self.parent_connection.send(
            MultiprocessRotaryEncoderAPI.Command(
                MultiprocessRotaryEncoderAPI.CommandFunction.WAIT_FOR_CART_TO_CROSS_CENTER,
                [
                    originally_left_of_center,
                    left_limit_degrees,
                    cart_mm_per_degree,
                    midline_mm
                ]
            )
        )

        return_value = self.parent_connection.recv()

        assert return_value is None

    def wait_for_termination(
            self
    ):

        self.parent_connection.send(
            MultiprocessRotaryEncoderAPI.Command(MultiprocessRotaryEncoderAPI.CommandFunction.TERMINATE)
        )

        return_value = self.parent_connection.recv()

        assert return_value is None

        self.process.join()


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
            right_limit_switch_input_pin: CkPin,
            max_timesteps_per_second: float,
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
        :param max_timesteps_per_second: Maximum timesteps per second.
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
        self.max_timesteps_per_second = max_timesteps_per_second

        self.midline_mm = self.limit_to_limit_mm / 2.0
        self.soft_limit_mm_from_midline = self.midline_mm - 100.0 - 45.0 / 2.0
        self.move_to_limit_motor_speed = 25
        self.move_away_from_limit_motor_speed = 15
        self.agent: Optional[MdpAgent] = None
        self.state_lock = RLock()
        self.previous_timestep_epoch: Optional[float] = None
        self.steps_per_second = 0.0

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

        self.cart_rotary_encoder = MultiprocessRotaryEncoderAPI(
            phase_a_pin=self.cart_rotary_encoder_phase_a_pin,
            phase_b_pin=self.cart_rotary_encoder_phase_b_pin
        )
        self.cart_rotary_encoder.wait_for_startup()
        self.cart_rotary_encoder_centered_state: Optional[Dict[str, float]] = None
        self.cart_rotary_encoder_left_limit_state: Optional[Dict[str, float]] = None
        self.cart_rotary_encoder_right_limit_state: Optional[Dict[str, float]] = None

        self.pole_rotary_encoder = MultiprocessRotaryEncoderAPI(
            phase_a_pin=self.pole_rotary_encoder_phase_a_pin,
            phase_b_pin=self.pole_rotary_encoder_phase_b_pin
        )
        self.pole_rotary_encoder.wait_for_startup()
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

        # calibration state and values
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

        # mark degrees at left and right limits. do this in whatever order is the most efficient given the cart's
        # current position. we can only do this after the initial calibration, since determining left of center
        # depends on having a value for the left limit.
        if self.left_limit_degrees is not None and self.is_left_of_center():
            self.left_limit_degrees = self.move_cart_to_left_limit()
            self.cart_rotary_encoder_left_limit_state = self.cart_rotary_encoder.capture_state()
            self.right_limit_degrees = self.move_cart_to_right_limit()
            self.cart_rotary_encoder_right_limit_state = self.cart_rotary_encoder.capture_state()
        else:
            self.right_limit_degrees = self.move_cart_to_right_limit()
            self.cart_rotary_encoder_right_limit_state = self.cart_rotary_encoder.capture_state()
            self.left_limit_degrees = self.move_cart_to_left_limit()
            self.cart_rotary_encoder_left_limit_state = self.cart_rotary_encoder.capture_state()

        # calibrate mm/degree and the midline
        self.limit_to_limit_degrees = abs(self.left_limit_degrees - self.right_limit_degrees)
        self.cart_mm_per_degree = self.limit_to_limit_mm / self.limit_to_limit_degrees
        self.midline_degrees = (self.left_limit_degrees + self.right_limit_degrees) / 2.0

        # center cart and capture initial conditions of the rotary encoders for resetting later
        self.center_cart(False)
        self.cart_rotary_encoder_centered_state = self.cart_rotary_encoder.capture_state()
        self.pole_rotary_encoder_stationary_state = self.pole_rotary_encoder.capture_state()
        self.pole_rotary_encoder_degrees_at_bottom = self.pole_rotary_encoder.get_degrees()

        logging.info(
            f'Calibrated:\n'
            f'\tLeft-limit degrees:  {self.left_limit_degrees}\n'
            f'\tRight-limit degrees:  {self.right_limit_degrees}\n'
            f'\tLimit-to-limit degrees:  {self.limit_to_limit_degrees}\n'
            f'\tCart mm / degree:  {self.cart_mm_per_degree}\n'
            f'\tMidline degree:  {self.midline_degrees}\n'
            f'\tPole degrees at bottom:  {self.pole_rotary_encoder_degrees_at_bottom}\n'
        )

    def move_cart_to_left_limit(
            self
    ) -> float:
        """
        Move the cart to the left limit.

        :return: Resulting degrees of rotation from the cart's rotary encoder.
        """

        if not self.left_limit_pressed.is_set():
            logging.info('Moving cart to the left and waiting for limit switch.')
            self.motor.set_speed(-self.move_to_limit_motor_speed)
            self.left_limit_pressed.wait()

        logging.info('Moving cart away from left limit switch.')
        self.motor.set_speed(self.move_away_from_limit_motor_speed)
        self.left_limit_released.wait()
        self.stop_cart()

        return self.cart_rotary_encoder.get_net_total_degrees()

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

            with self.state_lock:

                # it's important to stop the cart any time the limit switch is pressed
                self.stop_cart()

                # hitting a limit switch in the middle of an episode means that we've lost calibration. the soft limits
                # should have prevented this, but this failed. end the episode and calibrate upon the next episod reset.
                if self.state is not None and not self.state.terminal:
                    self.state = self.get_state(True)
                    self.calibrate_on_next_reset = True

            # another thread may attempt to wait for the switch to be released immediately upon the pressed event being
            # set. prevent a race condition by first clearing the released event before setting the pressed event.
            self.left_limit_released.clear()
            self.left_limit_pressed.set()

        else:

            logging.info('Left limit released.')

            # another thread may attempt to wait for the switch to be pressed immediately upon the released event being
            # set. prevent a race condition by first clearing the pressed event before setting the released event.
            self.left_limit_pressed.clear()
            self.left_limit_released.set()

    def move_cart_to_right_limit(
            self
    ) -> float:
        """
        Move the cart to the right limit.

        :return: Resulting degrees of rotation from the cart's rotary encoder.
        """

        if not self.right_limit_pressed.is_set():
            logging.info('Moving cart to the right and waiting for limit switch.')
            self.motor.set_speed(self.move_to_limit_motor_speed)
            self.right_limit_pressed.wait()

        logging.info('Moving cart away from right limit switch.')
        self.motor.set_speed(-self.move_away_from_limit_motor_speed)
        self.right_limit_released.wait()
        self.stop_cart()

        return self.cart_rotary_encoder.get_net_total_degrees()

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

            with self.state_lock:

                # it's important to stop the cart any time the limit switch is pressed
                self.stop_cart()

                # hitting a limit switch in the middle of an episode means that we've lost calibration. the soft limits
                # should have prevented this, but this failed. end the episode and calibrate upon the next episod reset.
                if self.state is not None and not self.state.terminal:
                    self.state = self.get_state(True)
                    self.calibrate_on_next_reset = True

            # another thread may attempt to wait for the switch to be released immediately upon the pressed event being
            # set. prevent a race condition by first clearing the released event before setting the pressed event.
            self.right_limit_released.clear()
            self.right_limit_pressed.set()

        else:

            logging.info('Right limit released.')

            # another thread may attempt to wait for the switch to be pressed immediately upon the released event being
            # set. prevent a race condition by first clearing the pressed event before setting the released event.
            self.right_limit_pressed.clear()
            self.right_limit_released.set()

    def center_cart(
            self,
            restore_limit_state: bool
    ):
        """
        Center the cart.

        :param restore_limit_state: Whether to restore the limit state before centering.
        """

        logging.info('Centering cart.')

        originally_left_of_center = self.is_left_of_center()

        # restore the limit state by physicially positioning the cart at the limit and restoring the state to what is
        # was when we calibrated. this corrects any loss of calibration that occurred while moving the cart. do this in
        # whatever order is the most efficient given the cart's current position.
        if restore_limit_state:
            if originally_left_of_center:
                self.move_cart_to_left_limit()
                self.cart_rotary_encoder.restore_captured_state(self.cart_rotary_encoder_left_limit_state)
            else:
                self.move_cart_to_right_limit()
                self.cart_rotary_encoder.restore_captured_state(self.cart_rotary_encoder_right_limit_state)

        # move toward the center, wait for the center to be reached, and stop the cart.
        self.motor.set_speed(
            self.move_to_limit_motor_speed if originally_left_of_center
            else -self.move_to_limit_motor_speed
        )
        self.cart_rotary_encoder.wait_for_cart_to_cross_center(
            originally_left_of_center,
            self.left_limit_degrees,
            self.cart_mm_per_degree,
            self.midline_mm
        )
        self.stop_cart()
        logging.info('Cart centered.\n')

        logging.info('Waiting for stationary pole.')
        self.pole_rotary_encoder.wait_for_stationarity()
        logging.info(f'Pole is stationary at degrees:  {self.pole_rotary_encoder.get_net_total_degrees():.1f}\n')

    def is_left_of_center(
            self,
    ) -> bool:
        """
        Check whether the cart is left of the midline.

        :return: True if the cart is left of the midline and False otherwise.
        """

        return abs(
            self.cart_rotary_encoder.get_net_total_degrees() - self.left_limit_degrees
        ) * self.cart_mm_per_degree < self.midline_mm

    def stop_cart(
            self
    ):
        """
        Stop the cart and do not return until it is stationary.
        """

        logging.info('Stopping cart...')
        self.motor.set_speed(0)
        self.cart_rotary_encoder.wait_for_stationarity()
        logging.info('Cart stopped.')

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

        # calibrate if needed, which leaves the cart centered.
        if self.calibrate_on_next_reset:
            self.calibrate()
            self.calibrate_on_next_reset = False

        # otherwise, center the cart with the current calibration and reset the rotary encoders to their calibration-
        # initial conditions.
        else:
            self.center_cart(True)
            self.cart_rotary_encoder.restore_captured_state(self.cart_rotary_encoder_centered_state)
            self.pole_rotary_encoder.restore_captured_state(self.pole_rotary_encoder_stationary_state)

        self.agent = agent
        self.state = self.get_state(False)
        self.previous_timestep_epoch = None
        self.steps_per_second = 0.0

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

                try:
                    self.motor.set_speed(self.motor.get_speed() + speed_change)
                except OSError as e:
                    logging.error(f'Error while setting speed:  {e}')

                self.state = self.get_state(None)

                if self.state.terminal:
                    self.stop_cart()

                reward_value = 1.0

            if self.previous_timestep_epoch is None:
                self.previous_timestep_epoch = time.time()
            else:
                current_timestep_epoch = time.time()
                steps_per_second = 1.0 / (current_timestep_epoch - self.previous_timestep_epoch)
                self.previous_timestep_epoch = current_timestep_epoch
                smoothing = 0.5
                self.steps_per_second = smoothing * self.steps_per_second + (1.0 - smoothing) * steps_per_second
                logging.debug(f'Running at {self.steps_per_second:.1f} steps/sec')

            time.sleep(1.0 / self.max_timesteps_per_second)

            logging.debug(f'State after step {t}:  {self.state}')

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

        mm_from_left_limit = abs(
            self.cart_rotary_encoder.get_net_total_degrees() - self.left_limit_degrees
        ) * self.cart_mm_per_degree

        mm_from_midline = mm_from_left_limit - self.limit_to_limit_mm / 2.0

        # terminate for violation of soft limit
        if terminal is None:
            terminal = abs(mm_from_midline) >= self.soft_limit_mm_from_midline
            if terminal:
                logging.info(
                    f'Cart position ({mm_from_midline:.1f}) mm exceeded soft limit '
                    f'({self.soft_limit_mm_from_midline}) mm. Terminating.'
                )

        return CartPoleState(
            environment=self,
            observation=np.array([
                mm_from_midline,
                self.cart_rotary_encoder.get_degrees_per_second() * self.cart_mm_per_degree,
                self.pole_rotary_encoder.get_degrees() - self.pole_rotary_encoder_degrees_at_bottom,
                self.pole_rotary_encoder.get_degrees_per_second()
            ]),
            agent=self.agent,
            terminal=terminal,
            truncated=False
        )

    def close(
            self
    ):
        """
        Close the environment, releasing resources.
        """

        self.cart_rotary_encoder.wait_for_termination()
        self.pole_rotary_encoder.wait_for_termination()
