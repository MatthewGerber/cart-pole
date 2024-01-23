import logging
import time
from argparse import ArgumentParser
from enum import Enum, auto
from multiprocessing import Process, Value, Pipe
# noinspection PyProtectedMember
from multiprocessing.connection import Connection
from threading import Event, RLock
from typing import List, Tuple, Any, Optional, Dict, Callable

import numpy as np
from numpy.random import RandomState
from smbus2 import SMBus

from raspberry_py.gpio import CkPin, get_ck_pin
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
            cart_mm_from_center: float,
            cart_velocity_mm_per_sec: float,
            pole_angle_deg_from_upright: float,
            pole_angular_velocity_deg_per_sec: float,
            agent: MdpAgent,
            terminal: bool,
            truncated: bool
    ):
        """
        Initialize the state.

        :param environment: Environment.
        :param cart_mm_from_center: Cart position as mm left (negative) or right (positive) from center.
        :param cart_velocity_mm_per_sec: Cart velocity (mm/sec).
        :param pole_angle_deg_from_upright: Pole angle as degrees left (negative) or right (positive) from upright.
        :param pole_angular_velocity_deg_per_sec: Pole angular velocity (deg/sec).
        :param agent: Agent.
        :param terminal: Whether the state is terminal, meaning the episode has terminated naturally due to the
        dynamics of the environment. For example, the natural dynamics of the environment terminate when the pole goes
        beyond the permitted bounds of the track.
        :param truncated: Whether the state is truncated, meaning the episode has ended for some reason other than the
        natural dynamics of the environment. For example, imposing an artificial time limit on an episode might cause
        the episode to end without the agent in a predefined goal state.
        """

        self.cart_mm_from_center = cart_mm_from_center
        self.cart_velocity_mm_per_second = cart_velocity_mm_per_sec
        self.pole_angle_deg_from_upright = pole_angle_deg_from_upright
        self.pole_angular_velocity_deg_per_sec = pole_angular_velocity_deg_per_sec

        self.observation = np.array([
            self.cart_mm_from_center,
            self.cart_velocity_mm_per_second,
            self.pole_angle_deg_from_upright,
            self.pole_angular_velocity_deg_per_sec
        ])

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
            f'cart pos={self.cart_mm_from_center:.1f} mm '
            f'@ {self.cart_velocity_mm_per_second:.1f} mm/s; '
            f'pole angle={self.pole_angle_deg_from_upright:.1f} deg '
            f'@ {self.pole_angular_velocity_deg_per_sec:.1f} deg/s'
        )


class MultiprocessRotaryEncoder:
    """
    Multiprocess wrapper around the rotary encoder. This allows the rotary encoder to receive phase-change events on a
    dedicated CPU core that is separate from the main program running the environment and RLAI. This is important
    because rotary encoders receive events at a high rate, and if events are dropped because the event callbacks are
    competing with other threads on the same core, then events can be dropped and the rotary encoder's output will be
    incorrect.
    """

    class SharedMemoryRotaryEncoder(RotaryEncoder):
        """
        Extension of the rotary encoder that provides shared-memory access to internal variables.
        """

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
                degrees_per_second_value: Value,
                clockwise_value: Value
        ):
            """
            Initialize the rotary encoder.

            :param phase_a_pin: Phase-a pin.
            :param phase_b_pin: Phase-b pin.
            :param phase_changes_per_rotation: Number of phase changes per rotation.
            :param report_state: A function from the current rotary encoder to a boolean indicating whether to report
            state when rotation changes. Because rotary encoders usually need to have very low latency, the added
            overhead of reporting state at ever phase change can reduce timeliness of the updates. Pass None to always
            report state.
            :param degrees_per_second_smoothing: Smoothing factor to apply to the degrees/second estimate, with 0.0
            being no smoothing (the new value equals the most recent value exactly), and 1.0 being complete smoothing
            (the new value equals the previous value exactly).
            :param bounce_time_ms: Bounce time (ms), or None for no value. This is not usually needed with high-quality
            rotary encoders that exhibit minimal mechanical bounce in their internal switches. Conversely, any nonzero
            bounce time will cause missed phase changes and inaccurate rotary encodings.
            :param net_total_degrees_value: Shared-memory structure for reading the current net-total degrees.
            :param degrees_value: Shared-memory structure for reading the current degrees.
            :param degrees_per_second_value: Shared-memory structure for reading the current degrees per second value.
            :param clockwise_value: Shared-memory structure for reading the clockwise value.
            """

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
            self.clockwise_value = clockwise_value

            self.set_shared_memory_values()

        def update_state(
                self
        ):
            """
            Update state.
            """

            super().update_state()

            self.set_shared_memory_values()

        def set_shared_memory_values(
                self
        ):
            """
            Set shared-memory values.
            """

            self.net_total_degrees_value.value = self.net_total_degrees
            self.degrees_value.value = self.degrees
            self.degrees_per_second_value.value = self.degrees_per_second
            self.clockwise_value.value = 1 if self.clockwise else 0

    class CommandFunction(Enum):
        """
        Command functions that can be sent to the rotary encoder.
        """

        # Wait for the rotary encoder process to fully start up.
        WAIT_FOR_STARTUP = auto()

        # Capture the state of the rotary encoder for subsequent restoration via RESTORE_CAPTURED_STATE.
        CAPTURE_STATE = auto()

        # Restore a state previously captured via CAPTURE_STATE.
        RESTORE_CAPTURED_STATE = auto()

        # Wait for the rotary encoder to become stationary.
        WAIT_FOR_STATIONARITY = auto()

        # Wait for the cart to cross the center of the track. This only applies to the cart's rotary encoder.
        WAIT_FOR_CART_TO_CROSS_CENTER = auto()

        # Terminate the process running the rotary encoder.
        TERMINATE = auto()

    class Command:
        """
        Command to send to the rotary encoder.
        """

        def __init__(
                self,
                function: 'MultiprocessRotaryEncoder.CommandFunction',
                args: Optional[List[Any]] = None
        ):
            """
            Initialize the command.

            :param function: Function.
            :param args: Optional list of arguments passed to the function.
            """

            if args is None:
                args = []

            self.function = function
            self.args = args

    @staticmethod
    def run_command_loop(
            identifier: str,
            phase_a_pin: CkPin,
            phase_b_pin: CkPin,
            degrees_per_second_smoothing: float,
            net_total_degrees_value: Value,
            degrees_value: Value,
            degrees_per_second_value: Value,
            clockwise_value: Value,
            command_pipe: Connection
    ):
        """
        Run the command loop. This instantiates the shared-memory rotary encoder, passing in shared-memory variables to
        enable reading the rotary encoder's state.

        :param identifier: Descriptive string identifier for the command loop.
        :param phase_a_pin: Phase-a pin.
        :param phase_b_pin: Phase-b pin.
        :param degrees_per_second_smoothing: Smoothing factor to apply to the degrees/second estimate, with 0.0 being no
        smoothing (the new value equals the most recent value exactly), and 1.0 being complete smoothing (the new value
        equals the previous value exactly).
        :param net_total_degrees_value: Shared-memory structure for reading the current net-total degrees.
        :param degrees_value: Shared-memory structure for reading the current degrees.
        :param degrees_per_second_value: Shared-memory structure for reading the current degrees per second value.
        :param clockwise_value: Shared-memory structure for reading the clockwise value.
        :param command_pipe: Command pipe that the command loop will use to receive commands and send return values.
        """

        rotary_encoder = MultiprocessRotaryEncoder.SharedMemoryRotaryEncoder(
            phase_a_pin=phase_a_pin,
            phase_b_pin=phase_b_pin,
            phase_changes_per_rotation=2400,
            report_state=lambda e: False,
            degrees_per_second_smoothing=degrees_per_second_smoothing,
            bounce_time_ms=None,
            net_total_degrees_value=net_total_degrees_value,
            degrees_value=degrees_value,
            degrees_per_second_value=degrees_per_second_value,
            clockwise_value=clockwise_value
        )

        while True:

            logging.info(f'{identifier}:  Waiting for command...')
            command: MultiprocessRotaryEncoder.Command = command_pipe.recv()
            logging.info(f'{identifier}:  Command received -- {command.function}')

            if command.function == MultiprocessRotaryEncoder.CommandFunction.WAIT_FOR_STARTUP:
                logging.info(f'{identifier}:  Startup complete.')
                return_value = None
            elif command.function == MultiprocessRotaryEncoder.CommandFunction.CAPTURE_STATE:
                return_value = rotary_encoder.capture_state()
            elif command.function == MultiprocessRotaryEncoder.CommandFunction.RESTORE_CAPTURED_STATE:
                rotary_encoder.restore_captured_state(*command.args)
                rotary_encoder.set_shared_memory_values()
                return_value = None
            elif command.function == MultiprocessRotaryEncoder.CommandFunction.WAIT_FOR_STATIONARITY:
                rotary_encoder.wait_for_stationarity(1.0)
                rotary_encoder.set_shared_memory_values()
                return_value = None
            elif command.function == MultiprocessRotaryEncoder.CommandFunction.WAIT_FOR_CART_TO_CROSS_CENTER:

                # unpack arguments
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

                # wait for a state event to be reported and set the thread-wait event
                center_reached = Event()
                center_reached_rpy_event = RpyEvent(lambda _: center_reached.set())
                rotary_encoder.events.append(center_reached_rpy_event)

                # wait for event
                center_reached.wait()

                # disable reporting and remove event
                rotary_encoder.report_state = lambda e: False
                rotary_encoder.events.remove(center_reached_rpy_event)

                return_value = None

            elif command.function == MultiprocessRotaryEncoder.CommandFunction.TERMINATE:
                logging.info(f'{identifier}:  Terminating.')
                break
            else:
                raise ValueError(f'Unknown function:  {command.function}')

            command_pipe.send(return_value)

        # we'll only get here when we break above
        command_pipe.send(None)

    def __init__(
            self,
            identifier: str,
            phase_a_pin: CkPin,
            phase_b_pin: CkPin,
            degrees_per_second_smoothing: float
    ):
        """
        Initialize the multiprocess rotary encoder.

        :param identifier: Descriptive string identifier for the process.
        :param phase_a_pin: Phase-a pin.
        :param phase_b_pin: Phase-b pin.
        :param degrees_per_second_smoothing: Smoothing factor to apply to the degrees/second estimate, with 0.0 being no
        smoothing (the new value equals the most recent value exactly), and 1.0 being complete smoothing (the new value
        equals the previous value exactly).
        """

        self.identifier = identifier
        self.phase_a_pin = phase_a_pin
        self.phase_b_pin = phase_b_pin
        self.degrees_per_second_smoothing = degrees_per_second_smoothing

        self.net_total_degrees_value = Value('d', 0.0)
        self.degrees_value = Value('d', 0.0)
        self.degrees_per_second_value = Value('d', 0.0)
        self.clockwise_value = Value('i', 0)
        self.parent_connection, self.child_connection = Pipe()
        self.process = Process(
            target=MultiprocessRotaryEncoder.run_command_loop,
            args=(
                self.identifier,
                self.phase_a_pin,
                self.phase_b_pin,
                self.degrees_per_second_smoothing,
                self.net_total_degrees_value,
                self.degrees_value,
                self.degrees_per_second_value,
                self.clockwise_value,
                self.child_connection
            )
        )
        self.process.start()

    def get_net_total_degrees(
            self
    ) -> float:
        """
        Get net total degrees in (-inf,inf).

        :return: Degrees.
        """

        return self.net_total_degrees_value.value

    def get_degrees(
            self
    ) -> float:
        """
        Get rotational degrees in [0,360].

        :return: Degrees.
        """

        return self.degrees_value.value

    def get_degrees_per_second(
            self
    ) -> float:
        """
        Get degrees per second.

        :return: Degrees per second.
        """

        return self.degrees_per_second_value.value

    def get_clockwise(
            self
    ) -> bool:
        """
        Get clockwise.

        :return: True if clockwise.
        """

        return self.clockwise_value.value == 1

    def wait_for_startup(
            self
    ):
        """
        Wait for startup.
        """

        self.parent_connection.send(
            MultiprocessRotaryEncoder.Command(MultiprocessRotaryEncoder.CommandFunction.WAIT_FOR_STARTUP)
        )

        return_value = self.parent_connection.recv()

        assert return_value is None

    def capture_state(
            self
    ) -> Dict[str, float]:
        """
        Capture state.

        :return: Captured state.
        """

        self.parent_connection.send(
            MultiprocessRotaryEncoder.Command(MultiprocessRotaryEncoder.CommandFunction.CAPTURE_STATE)
        )

        return self.parent_connection.recv()

    def restore_captured_state(
            self,
            captured_state: Dict[str, float]
    ):
        """
        Restore captured state.

        :param captured_state: Captured state.
        """

        self.parent_connection.send(
            MultiprocessRotaryEncoder.Command(
                MultiprocessRotaryEncoder.CommandFunction.RESTORE_CAPTURED_STATE,
                [captured_state]
            )
        )

        return_value = self.parent_connection.recv()

        assert return_value is None

    def wait_for_stationarity(
            self
    ):
        """
        Wait for stationarity.
        """

        self.parent_connection.send(
            MultiprocessRotaryEncoder.Command(MultiprocessRotaryEncoder.CommandFunction.WAIT_FOR_STATIONARITY)
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
        """
        Wait for the cart to cross the center of the track.

        :param originally_left_of_center: Whether the cart was originally left of center.
        :param left_limit_degrees: Left-limit degrees.
        :param cart_mm_per_degree: Cart's calibrated mm/degree.
        :param midline_mm: Track midline (mm).
        """

        self.parent_connection.send(
            MultiprocessRotaryEncoder.Command(
                MultiprocessRotaryEncoder.CommandFunction.WAIT_FOR_CART_TO_CROSS_CENTER,
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
        """
        Wait for termination.
        """

        self.parent_connection.send(
            MultiprocessRotaryEncoder.Command(MultiprocessRotaryEncoder.CommandFunction.TERMINATE)
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
            '--motor-negative-speed-is-left',
            type=bool,
            default='true',
            action='store_false',
            help='Whether negative motor speed moves the cart to the left.'
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
            '--cart-rotary-encoder-phase-b-pin',
            type=get_ck_pin,
            help=(
                'GPIO pin connected to the phase-b input of the cart\'s rotary encoder. This can be an enumerated '
                'type and name from either the raspberry_py.gpio.Pin class (e.g., Pin.GPIO_5) or the '
                'raspberry_py.gpio.CkPin class (e.g., CkPin.GPIO5).'
            )
        )

        parser.add_argument(
            '--pole-rotary-encoder-phase-a-pin',
            type=get_ck_pin,
            help=(
                'GPIO pin connected to the phase-a input of the pole\'s rotary encoder. This can be an enumerated '
                'type and name from either the raspberry_py.gpio.Pin class (e.g., Pin.GPIO_5) or the '
                'raspberry_py.gpio.CkPin class (e.g., CkPin.GPIO5).'
            )
        )

        parser.add_argument(
            '--pole-rotary-encoder-phase-b-pin',
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
            soft_limit_standoff_mm: float,
            cart_width_mm: float,
            motor_pwm_channel: int,
            motor_pwm_direction_pin: CkPin,
            motor_negative_speed_is_left: bool,
            cart_rotary_encoder_phase_a_pin: CkPin,
            cart_rotary_encoder_phase_b_pin: CkPin,
            pole_rotary_encoder_phase_a_pin: CkPin,
            pole_rotary_encoder_phase_b_pin: CkPin,
            left_limit_switch_input_pin: CkPin,
            right_limit_switch_input_pin: CkPin,
            timesteps_per_second: float
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
        :param motor_negative_speed_is_left: Whether negative motor speeds move the cart to the left.
        :param cart_rotary_encoder_phase_a_pin: Cart rotary encoder phase-a pin.
        :param cart_rotary_encoder_phase_b_pin: Cart rotary encoder phase-b pin.
        :param pole_rotary_encoder_phase_a_pin: Pole rotary encoder phase-a pin.
        :param pole_rotary_encoder_phase_b_pin: Pole rotary encoder phase-b pin.
        :param left_limit_switch_input_pin: Left limit pin.
        :param right_limit_switch_input_pin: Right limit pin.
        :param timesteps_per_second: Number of environment advancement steps to execute per second.
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
        self.motor_negative_speed_is_left = motor_negative_speed_is_left
        self.cart_rotary_encoder_phase_a_pin = cart_rotary_encoder_phase_a_pin
        self.cart_rotary_encoder_phase_b_pin = cart_rotary_encoder_phase_b_pin
        self.pole_rotary_encoder_phase_a_pin = pole_rotary_encoder_phase_a_pin
        self.pole_rotary_encoder_phase_b_pin = pole_rotary_encoder_phase_b_pin
        self.left_limit_switch_input_pin = left_limit_switch_input_pin
        self.right_limit_switch_input_pin = right_limit_switch_input_pin
        self.timesteps_per_second = timesteps_per_second

        self.midline_mm = self.limit_to_limit_mm / 2.0
        self.soft_limit_mm_from_midline = self.midline_mm - self.soft_limit_standoff_mm - self.cart_width_mm / 2.0
        self.move_to_limit_motor_speed = 25
        self.move_away_from_limit_motor_speed = 15
        self.agent: Optional[MdpAgent] = None
        self.state_lock = RLock()
        self.previous_timestep_epoch: Optional[float] = None
        self.current_timesteps_per_second = 0.0
        self.current_timesteps_per_second_smoothing = 0.75
        self.timestep_sleep_seconds = 1.0 / self.timesteps_per_second

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

        self.cart_rotary_encoder = MultiprocessRotaryEncoder(
            identifier='cart-rotary-encoder',
            phase_a_pin=self.cart_rotary_encoder_phase_a_pin,
            phase_b_pin=self.cart_rotary_encoder_phase_b_pin,
            degrees_per_second_smoothing=0.95,
        )
        self.cart_rotary_encoder.wait_for_startup()
        self.cart_rotary_encoder_state_at_center: Optional[Dict[str, float]] = None
        self.cart_rotary_encoder_state_at_left_limit: Optional[Dict[str, float]] = None
        self.cart_rotary_encoder_state_at_right_limit: Optional[Dict[str, float]] = None

        self.pole_rotary_encoder = MultiprocessRotaryEncoder(
            identifier='pole-rotary-encoder',
            phase_a_pin=self.pole_rotary_encoder_phase_a_pin,
            phase_b_pin=self.pole_rotary_encoder_phase_b_pin,
            degrees_per_second_smoothing=0.95,
        )
        self.pole_rotary_encoder.wait_for_startup()
        self.pole_rotary_encoder_state_at_bottom: Optional[Dict[str, float]] = None
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
        self.minimum_motor_speed_left: Optional[int] = None
        self.minimum_motor_speed_right: Optional[int] = None

    def calibrate(
            self
    ):
        """
        Calibrate the cart-pole apparatus, leaving the cart centered.
        """

        logging.info('Calibrating.')

        # mark degrees at left and right limits. do this in whatever order is the most efficient given the cart's
        # current position. we can only do this efficiency trick after the initial calibration, since determining left
        # of center depends on having a value for the left limit. regardless, capture the rotary encoder's state at the
        # right and left limits for subsequent restoration.
        if self.left_limit_degrees is not None and self.is_left_of_center():
            self.left_limit_degrees = self.move_cart_to_left_limit()
            self.cart_rotary_encoder_state_at_left_limit = self.cart_rotary_encoder.capture_state()
            self.right_limit_degrees = self.move_cart_to_right_limit()
            self.cart_rotary_encoder_state_at_right_limit = self.cart_rotary_encoder.capture_state()
        else:
            self.right_limit_degrees = self.move_cart_to_right_limit()
            self.cart_rotary_encoder_state_at_right_limit = self.cart_rotary_encoder.capture_state()
            self.left_limit_degrees = self.move_cart_to_left_limit()
            self.cart_rotary_encoder_state_at_left_limit = self.cart_rotary_encoder.capture_state()

        # calibrate mm/degree and the midline
        self.limit_to_limit_degrees = abs(self.left_limit_degrees - self.right_limit_degrees)
        self.cart_mm_per_degree = self.limit_to_limit_mm / self.limit_to_limit_degrees
        self.midline_degrees = (self.left_limit_degrees + self.right_limit_degrees) / 2.0

        # center cart and capture initial conditions of the rotary encoders for subsequent restoration. we captured the
        # limit state above, and the cart hasn't moved. and we're about to center the cart and captured the center
        # state. so, no need to restore either of these.
        self.center_cart(False, False)
        self.cart_rotary_encoder_state_at_center = self.cart_rotary_encoder.capture_state()
        self.pole_rotary_encoder_state_at_bottom = self.pole_rotary_encoder.capture_state()
        self.pole_rotary_encoder_degrees_at_bottom = self.pole_rotary_encoder.get_degrees()

        # identify the minimum motor speeds that will get the cart to move left and right. there's a deadzone in the
        # middle that depends on the logic of the motor circuitry, mass and friction of the assembly, etc. this
        # nonlinearity of motor speed and cart velocity will confuse the controller.
        state = self.get_state(-1, True)
        self.minimum_motor_speed_left = 0
        while state.cart_velocity_mm_per_second > -15.0:
            self.minimum_motor_speed_left -= 1
            self.motor.set_speed(self.minimum_motor_speed_left)
            time.sleep(0.5)
            state = self.get_state(-1, True)
            logging.debug(f'Velocity:  {state.cart_velocity_mm_per_second:.2f} mm/sec')
        self.stop_cart()
        state = self.get_state(-1, True)
        self.minimum_motor_speed_right = 0
        while state.cart_velocity_mm_per_second < 15.0:
            self.minimum_motor_speed_right += 1
            self.motor.set_speed(self.minimum_motor_speed_right)
            time.sleep(0.5)
            state = self.get_state(-1, True)
            logging.debug(f'Velocity:  {state.cart_velocity_mm_per_second:.2f} mm/sec')
        self.stop_cart()

        # recent the cart and restore the center state. the limit states should be fine.
        self.center_cart(False, True)

        logging.info(
            f'Calibrated:\n'
            f'\tLeft-limit degrees:  {self.left_limit_degrees}\n'
            f'\tRight-limit degrees:  {self.right_limit_degrees}\n'
            f'\tLimit-to-limit degrees:  {self.limit_to_limit_degrees}\n'
            f'\tCart mm / degree:  {self.cart_mm_per_degree}\n'
            f'\tMidline degree:  {self.midline_degrees}\n'
            f'\tPole degrees at bottom:  {self.pole_rotary_encoder_degrees_at_bottom}\n'
            f'\tMinimum motor speed left:  {self.minimum_motor_speed_left}\n'
            f'\tMinimum motor speed right:  {self.minimum_motor_speed_right}\n'
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

        self.handle_limit_event(
            'Left',
            is_pressed,
            self.left_limit_pressed,
            self.left_limit_released
        )

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
                # should have prevented this, but this failed. end the episode and calibrate upon the next episod reset.
                if self.state is not None and not self.state.terminal:
                    self.state = self.get_state(-1, True)
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
        is somewhat expensive, since it involves physically moving the cart to a limit switch.
        :param restore_center_state: Whether to restore the center state after centering. This ensures that the initial
        movements away from the center will be equivalent to previous such movements.
        at that position.
        """

        logging.info('Centering cart.')

        originally_left_of_center = self.is_left_of_center()

        # restore the limit state by physicially positioning the cart at the limit and restoring the state to what is
        # was when we calibrated. this corrects any loss of calibration that occurred while moving the cart. do this in
        # whatever order is the most efficient given the cart's current position.
        if restore_limit_state:
            if originally_left_of_center:
                self.move_cart_to_left_limit()
                self.cart_rotary_encoder.restore_captured_state(self.cart_rotary_encoder_state_at_left_limit)
            else:
                self.move_cart_to_right_limit()
                self.cart_rotary_encoder.restore_captured_state(self.cart_rotary_encoder_state_at_right_limit)

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

        # restore the center state
        if restore_center_state:
            self.cart_rotary_encoder.restore_captured_state(self.cart_rotary_encoder_state_at_center)
            self.pole_rotary_encoder.restore_captured_state(self.pole_rotary_encoder_state_at_bottom)

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
        Reset the environment to its initial conditions.

        :param agent: Agent used to generate on-the-fly state identifiers.
        :return: Initial state.
        """

        logging.info(f'Reset {self.num_resets + 1}.')

        super().reset_for_new_run(self.agent)

        self.agent = agent
        self.motor.start()

        # calibrate if needed, which leaves the cart centered in its initial conditions with the state captured.
        if self.calibrate_on_next_reset:
            self.calibrate()
            self.calibrate_on_next_reset = False

        # otherwise, center the cart with the current calibration and reset the rotary encoders to their calibration-
        # initial conditions.
        else:
            self.center_cart(True, True)

        self.state = self.get_state(0, False)
        self.previous_timestep_epoch = None
        self.current_timesteps_per_second = 0.0

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
                next_speed = self.motor.get_speed() + speed_change

                # if the next speed falls into the motor's dead zone, bump it to the minimum speed based on the
                # direction of speed change.
                if self.minimum_motor_speed_left < next_speed < self.minimum_motor_speed_right:
                    if speed_change < 0:
                        next_speed = self.minimum_motor_speed_left
                    elif speed_change > 0:
                        next_speed = self.minimum_motor_speed_right

                # we occasionally get i/o errors from the underlying interface to the pwm. this probably has something
                # to do with attempting to write new values at a high rate like we're doing here. catch any such error
                # and try again next time.
                try:
                    self.motor.set_speed(next_speed)
                except OSError as e:
                    logging.error(f'Error while setting speed:  {e}')

                self.state = self.get_state(t, None)

                if self.state.terminal:
                    self.stop_cart()

                reward_value = np.exp(
                    -(
                        np.abs([
                            self.state.cart_mm_from_center,
                            self.state.pole_angle_deg_from_upright
                        ]).sum()
                    )
                )

            if self.previous_timestep_epoch is None:
                self.previous_timestep_epoch = time.time()
            else:

                # update current timesteps per second
                current_timestep_epoch = time.time()
                current_timesteps_per_second = 1.0 / (current_timestep_epoch - self.previous_timestep_epoch)
                self.previous_timestep_epoch = current_timestep_epoch
                self.current_timesteps_per_second = (
                    self.current_timesteps_per_second_smoothing * self.current_timesteps_per_second +
                    (1.0 - self.current_timesteps_per_second_smoothing) * current_timesteps_per_second
                )

                # adapt the timestep sleep duration to achieve the target steps per second, given the overhead involved
                # in executing each call to advance.
                if self.current_timesteps_per_second > self.timesteps_per_second:
                    self.timestep_sleep_seconds *= 1.01
                elif self.current_timesteps_per_second < self.timesteps_per_second:
                    self.timestep_sleep_seconds *= 0.99

                logging.debug(f'Running at {self.current_timesteps_per_second:.1f} steps/sec')

            time.sleep(self.timestep_sleep_seconds)

            logging.debug(f'State after step {t}:  {self.state}')

            return self.state, Reward(None, reward_value)

    def get_state(
            self,
            t: int,
            terminal: Optional[bool]
    ) -> CartPoleState:
        """
        Get the current state.

        :param t: Time step.
        :param terminal: Whether to force a terminal state, or None for natural assessment.
        :return: State.
        """

        cart_mm_from_left_limit = abs(
            self.cart_rotary_encoder.get_net_total_degrees() - self.left_limit_degrees
        ) * self.cart_mm_per_degree

        cart_mm_from_center = cart_mm_from_left_limit - self.limit_to_limit_mm / 2.0

        # terminate for violation of soft limit
        if terminal is None:
            terminal = abs(cart_mm_from_center) >= self.soft_limit_mm_from_midline
            if terminal:
                logging.info(
                    f'Cart position ({cart_mm_from_center:.1f}) mm exceeded soft limit '
                    f'({self.soft_limit_mm_from_midline}) mm. Terminating.'
                )

        # get pole's degree from vertical at bottom
        pole_angle_deg_from_bottom = self.pole_rotary_encoder.get_degrees() - self.pole_rotary_encoder_degrees_at_bottom

        # translate to [-180,180] degrees from bottom. if the degrees value is beyond these bounds, then subtract a
        # complete rotation in the appropriate direction in order to get the same degree orientation but within the
        # desired bounds.
        if abs(pole_angle_deg_from_bottom) > 180.0:
            pole_angle_deg_from_bottom -= np.sign(pole_angle_deg_from_bottom) * 360.0

        # convert to degrees from upright, which is what we need for the reward calculation.
        if pole_angle_deg_from_bottom == 0.0:
            pole_angle_deg_from_upright = 180.0  # equivalent to -180.0
        else:
            pole_angle_deg_from_upright = np.sign(pole_angle_deg_from_bottom) * 180.0 - pole_angle_deg_from_bottom

        return CartPoleState(
            environment=self,
            cart_mm_from_center=cart_mm_from_center,
            cart_velocity_mm_per_sec=(
                -self.cart_rotary_encoder.get_degrees_per_second() * self.cart_mm_per_degree
            ),
            pole_angle_deg_from_upright=pole_angle_deg_from_upright,
            pole_angular_velocity_deg_per_sec=self.pole_rotary_encoder.get_degrees_per_second(),
            agent=self.agent,
            terminal=terminal,
            truncated=self.T is not None and t >= self.T
        )

    def close(
            self
    ):
        """
        Close the environment, releasing resources.
        """

        self.cart_rotary_encoder.wait_for_termination()
        self.pole_rotary_encoder.wait_for_termination()
