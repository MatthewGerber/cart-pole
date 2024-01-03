import time
from threading import Event
from typing import List, Tuple, Any

from numpy.random import RandomState
from smbus2 import SMBus

from raspberry_py.gpio import CkPin
from raspberry_py.gpio.controls import LimitSwitch
from raspberry_py.gpio.integrated_circuits import PulseWaveModulatorPCA9685PW
from raspberry_py.gpio.motors import DcMotor, DcMotorDriverIndirectPCA9685PW
from raspberry_py.gpio.sensors import RotaryEncoder
from rlai.core import MdpState, Action, Agent, Reward, Environment
from rlai.core.environments.mdp import MdpEnvironment


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
            '--limit-to-limit-distance-mm',
            type=float,
            help='The distance from one end (limit) to the other end (limit) of the cart-pole limit switches.'
        )

        parser.add_argument(
            '--inside-limits-motor-pwm-channel',
            type=int,
            help=(
                'Pulse-wave modulation (PWM) channel to use for motor control when the cart is operating inside the '
                'bounds of the limit switches.'
            )
        )

        parser.add_argument(
            '--outside-limits-motor-pwm-channel',
            type=int,
            help=(
                'Pulse-wave modulation (PWM) channel to use for motor control when the cart is operating outside the '
                'bounds of the limit switches.'
            )
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
            '--motor-side-limit-switch-input-pin',
            type=str,
            help=(
                'GPIO pin connected to the input pin of the motor-side limit switch. This can be an enumerated '
                'name and value from either the raspberry_py.gpio.Pin class (e.g., Pin.GPIO_5) or the '
                'raspberry_py.gpio.CkPin class (e.g., CkPin.GPIO5).'
            )
        )

        parser.add_argument(
            '--rotary-encoder-side-limit-switch-input-pin',
            type=str,
            help=(
                'GPIO pin connected to the input pin of the rotary-encoder-side limit switch. This can be an '
                'enumerated name and value from either the raspberry_py.gpio.Pin class (e.g., Pin.GPIO_5) or the '
                'raspberry_py.gpio.CkPin class (e.g., CkPin.GPIO5).'
            )
        )

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
            limit_to_limit_distance_mm: float,
            inside_limits_motor_pwm_channel: int,
            outside_limits_motor_pwm_channel: int,
            motor_pwm_direction_pin: CkPin,
            motor_negative_speed_is_left: bool,
            cart_rotary_encoder_phase_a_pin: CkPin,
            cart_rotary_encoder_phase_b_pin: CkPin,
            pole_rotary_encoder_phase_a_pin: CkPin,
            pole_rotary_encoder_phase_b_pin: CkPin,
            motor_side_limit_switch_input_pin: CkPin,
            rotary_encoder_side_limit_switch_input_pin: CkPin
    ):
        """
        Initialize the cart-pole environment.

        :param name: Name.
        :param random_state: Random state.
        :param T: Maximum number of steps to run, or None for no limit.
        :param limit_to_limit_distance_mm: The distance from one end (limit) to the other end (limit) of the cart-pole
        limit switches.
        :param inside_limits_motor_pwm_channel: Pulse-wave modulation (PWM) channel to use for motor control when the
        cart is operating inside the bounds of the limit switches.
        :param outside_limits_motor_pwm_channel: Pulse-wave modulation (PWM) channel to use for motor control when the
        cart is operating outside the bounds of the limit switches.
        :param motor_pwm_direction_pin: Motor's PWM direction pin.
        :param motor_negative_speed_is_left: Whether negative motor speeds move the cart to the left.
        :param cart_rotary_encoder_phase_a_pin: Cart rotary encoder phase-a pin.
        :param cart_rotary_encoder_phase_b_pin: Cart rotary encoder phase-b pin.
        :param pole_rotary_encoder_phase_a_pin: Pole rotary encoder phase-a pin.
        :param pole_rotary_encoder_phase_b_pin: Pole rotary encoder phase-b pin.
        :param motor_side_limit_switch_input_pin: Motor-side limit pin.
        :param rotary_encoder_side_limit_switch_input_pin: Rotary-encoder-side limit pin.
        """

        super().__init__(
            name=name,
            random_state=random_state,
            T=T
        )

        self.limit_to_limit_distance_mm = limit_to_limit_distance_mm
        self.midline_mm = self.limit_to_limit_distance_mm / 2.0
        self.inside_limits_motor_pwm_channel = inside_limits_motor_pwm_channel
        self.outside_limits_motor_pwm_channel = outside_limits_motor_pwm_channel
        self.motor_pwm_direction_pin = motor_pwm_direction_pin
        self.motor_negative_speed_is_left = motor_negative_speed_is_left
        self.cart_rotary_encoder_phase_a_pin = cart_rotary_encoder_phase_a_pin
        self.cart_rotary_encoder_phase_b_pin = cart_rotary_encoder_phase_b_pin
        self.pole_rotary_encoder_phase_a_pin = pole_rotary_encoder_phase_a_pin
        self.pole_rotary_encoder_phase_b_pin = pole_rotary_encoder_phase_b_pin
        self.motor_side_limit_switch_input_pin = motor_side_limit_switch_input_pin
        self.rotary_encoder_side_limit_switch_input_pin = rotary_encoder_side_limit_switch_input_pin

        self.actions = [
            ContinuousMultiDimensionalAction(
                value=None,
                min_values=np.array([-5.0]),
                max_values=np.array([5.0])
            )
        ]

        self.cart_rotary_encoder = RotaryEncoder(
            phase_a_pin=self.cart_rotary_encoder_phase_a_pin,
            phase_b_pin=self.cart_rotary_encoder_phase_b_pin,
            phase_changes_per_rotation=2400,
            report_state=False
        )

        self.pole_rotary_encoder = RotaryEncoder(
            phase_a_pin=self.pole_rotary_encoder_phase_a_pin,
            phase_b_pin=self.pole_rotary_encoder_phase_b_pin,
            phase_changes_per_rotation=2400,
            report_state=False
        )
        self.pole_rotary_encoder_degrees_at_bottom: Optional[float] = None

        self.pca9685pw = PulseWaveModulatorPCA9685PW(
            bus=SMBus('/dev/i2c-1'),
            address=PulseWaveModulatorPCA9685PW.PCA9685PW_ADDRESS,
            frequency_hz=500
        )

        self.inside_limits_motor_controller = DcMotor(
            driver=DcMotorDriverIndirectPCA9685PW(
                pca9685pw=self.pca9685pw,
                pwm_channel=self.inside_limits_motor_pwm_channel,
                direction_pin=self.motor_pwm_direction_pin,
                reverse=not self.motor_negative_speed_is_left
            ),
            speed=0
        )

        self.outside_limits_motor_controller = DcMotor(
            driver=DcMotorDriverIndirectPCA9685PW(
                pca9685pw=self.pca9685pw,
                pwm_channel=self.outside_limits_motor_pwm_channel,
                direction_pin=self.motor_pwm_direction_pin,
                reverse=not self.motor_negative_speed_is_left
            ),
            speed=0
        )

        self.motor_side_limit_switch = LimitSwitch(
            input_pin=self.motor_side_limit_switch_input_pin,
            bounce_time_ms=5
        )
        self.motor_side_limit_pressed = Event()
        self.motor_side_limit_released = Event()
        self.motor_side_limit_degrees: Optional[int] = None
        self.motor_side_limit_switch.event(lambda s: self.motor_side_limit_event(s.is_pressed()))

        self.rotary_encoder_side_limit_switch = LimitSwitch(
            input_pin=self.rotary_encoder_side_limit_switch_input_pin,
            bounce_time_ms=5
        )
        self.rotary_encoder_side_limit_pressed = Event()
        self.rotary_encoder_side_limit_released = Event()
        self.rotary_encoder_side_limit_degrees: Optional[int] = None
        self.rotary_encoder_side_limit_switch.event(lambda s: self.rotary_encoder_side_limit_event(s.is_pressed()))

        self.cart_mm_per_degree: Optional[float] = None
        self.cart_degrees_per_mm: Optional[float] = None
        self.limit_to_limit_degrees: Optional[float] = None
        self.midline_degrees: Optional[float] = None

    def move_cart_to_motor_side_limit(
            self
    ):
        """
        Move the cart to the motor-side limit.
        """

        if not self.motor_side_limit_pressed.is_set():
            self.inside_limits_motor_controller.set_speed(-5)
            self.motor_side_limit_pressed.wait()

        self.outside_limits_motor_controller.set_speed(1)
        self.motor_side_limit_released.wait()

    def motor_side_limit_event(
            self,
            is_pressed: bool
    ):
        """
        Receive a limit event from the motor side.

        :param is_pressed: Whether the limit switch is pressed.
        """

        if is_pressed:
            self.inside_limits_motor_controller.set_speed(0)
            self.motor_side_limit_released.clear()
            self.motor_side_limit_pressed.set()
        else:
            self.motor_side_limit_degrees = self.cart_rotary_encoder.degrees
            self.outside_limits_motor_controller.set_speed(0)
            self.motor_side_limit_pressed.clear()
            self.motor_side_limit_released.set()

    def move_cart_to_rotary_encoder_side_limit(
            self
    ):
        """
        Move the cart to the rotary-encoder-side limit.
        :return:
        """

        if not self.rotary_encoder_side_limit_pressed.is_set():
            self.inside_limits_motor_controller.set_speed(5)
            self.rotary_encoder_side_limit_pressed.wait()

        self.outside_limits_motor_controller.set_speed(-1)
        self.rotary_encoder_side_limit_released.wait()

    def rotary_encoder_side_limit_event(
            self,
            is_pressed: bool
    ):
        """
        Receive a limit event from the rotary encoder side.

        :param is_pressed: Whether the limit switch is pressed.
        """

        if is_pressed:
            self.inside_limits_motor_controller.set_speed(0)
            self.rotary_encoder_side_limit_released.clear()
            self.rotary_encoder_side_limit_pressed.set()
        else:
            self.rotary_encoder_side_limit_degrees = self.cart_rotary_encoder.degrees
            self.outside_limits_motor_controller.set_speed(0)
            self.rotary_encoder_side_limit_pressed.clear()
            self.rotary_encoder_side_limit_released.set()

    def calibrate(
            self
    ):
        """
        Calibrate the cart-pole apparatus.
        """

        self.move_cart_to_motor_side_limit()
        self.move_cart_to_rotary_encoder_side_limit()
        self.limit_to_limit_degrees = abs(self.motor_side_limit_degrees - self.rotary_encoder_side_limit_degrees)
        self.cart_mm_per_degree = self.limit_to_limit_distance_mm / self.limit_to_limit_degrees
        self.cart_degrees_per_mm = self.limit_to_limit_degrees / self.limit_to_limit_distance_mm
        self.midline_degrees = self.motor_side_limit_degrees + self.limit_to_limit_degrees / 2.0

    def center_cart(
            self
    ):
        """
        Center the cart.
        """

        self.move_cart_to_motor_side_limit()
        self.cart_rotary_encoder.report_state = True
        self.inside_limits_motor_controller.set_speed(5)
        self.cart_rotary_encoder.event(lambda s: (
            self.inside_limits_motor_controller.set_speed(0)
            if abs(s.degrees - self.motor_side_limit_degrees) / self.cart_degrees_per_mm >= self.midline_mm
            else None
        ))
        self.cart_rotary_encoder.report_state = False

        previous_pole_num_phase_changes = self.pole_rotary_encoder.num_phase_changes
        time.sleep(1.0)
        while self.pole_rotary_encoder.num_phase_changes != previous_pole_num_phase_changes:
            time.sleep(1.0)

        self.pole_rotary_encoder_degrees_at_bottom = self.pole_rotary_encoder.net_total_degrees

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

        return self.create_state(agent)

    def advance(
            self,
            state: MdpState,
            t: int,
            a: Action,
            agent: Agent
    ) -> Tuple[MdpState, Reward]:
        pass

    def create_state(
            self,
            agent: Any
    ) -> CartPoleState:

        return CartPoleState(
            environment=self,
            observation=np.array([
                (
                    abs(self.cart_rotary_encoder.net_total_degrees - self.motor_side_limit_degrees) *
                    self.cart_mm_per_degree
                ),
                self.cart_rotary_encoder.degrees_per_second * self.cart_mm_per_degree,
                (
                    self.pole_rotary_encoder.net_total_degrees - self.pole_rotary_encoder_degrees_at_bottom % 360.0
                ),
                self.pole_rotary_encoder.degrees_per_second
            ]),
            agent=agent,
            terminal=False,
            truncated=False
        )
