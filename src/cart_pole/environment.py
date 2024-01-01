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


class CartPole(MdpEnvironment):
    """
    Cart-pole environment for the Raspberry Pi.
    """

    @classmethod
    def init_from_arguments(
            cls,
            args: List[str],
            random_state: RandomState
    ) -> Tuple[Environment, List[str]]:
        pass

    def __init__(
            self,
            limit_to_limit_distance_mm: float,
            inside_limits_motor_pwm_channel: int,
            outside_limits_motor_pwm_channel: int
    ):
        super().__init__(
            name=name,
            random_state=rando_state,
            T=T
        )

        self.limit_to_limit_distance_mm = limit_to_limit_distance_mm

        self.midline_mm = self.limit_to_limit_distance_mm / 2.0

        self.pole_rotary_encoder = RotaryEncoder(
            phase_a_pin=pole_rotary_encoder_phase_a_pin,
            phase_b_pin=pole_rotary_encoder_phase_b_pin,
            phase_changes_per_rotation=2400,
            report_state=False
        )

        self.cart_rotary_encoder = RotaryEncoder(
            phase_a_pin=cart_rotary_encoder_phase_a_pin,
            phase_b_pin=cart_rotary_encoder_phase_b_pin,
            phase_changes_per_rotation=2400,
            report_state=False
        )

        self.pca9685pw = PulseWaveModulatorPCA9685PW(
            bus=SMBus('/dev/i2c-1'),
            address=PulseWaveModulatorPCA9685PW.PCA9685PW_ADDRESS,
            frequency_hz=400
        )

        self.inside_limits_motor_controller = DcMotor(
            driver=DcMotorDriverIndirectPCA9685PW(
                pca9685pw=self.pca9685pw,
                pwm_channel=inside_limits_motor_pwm_channel,
                direction_pin=CkPin.GPIO21
            ),
            speed=0
        )

        self.outside_limits_motor_controller = DcMotor(
            driver=DcMotorDriverIndirectPCA9685PW(
                pca9685pw=self.pca9685pw,
                pwm_channel=outside_limits_motor_pwm_channel,
                direction_pin=CkPin.GPIO21
            ),
            speed=0
        )

        self.motor_side_limit_switch = LimitSwitch(
            input_pin=motor_side_limit_switch_input_pin,
            bounce_time_ms=5
        )
        self.motor_side_limit_switch.event(lambda s: self.motor_side_limit_event(s.is_pressed()))
        self.motor_side_limit_pressed = Event()
        self.motor_side_limit_released = Event()

        self.rotary_encoder_side_limit_switch = LimitSwitch(
            input_pin=rotary_encoder_side_limit_switch_input_pin,
            bounce_time_ms=5
        )
        self.rotary_encoder_side_limit_switch.event(lambda s: self.rotary_encoder_side_limit_event(s.is_pressed()))
        self.rotary_encoder_side_limit_pressed = Event()
        self.rotary_encoder_side_limit_released = Event()

        self.degrees_at_motor_side_limit: Optional[int] = None
        self.degrees_at_rotary_encoder_side_limit: Optional[int] = None
        self.cart_mm_per_degree: Optional[float] = None
        self.limit_to_limit_degrees: Optional[int] = None

    def move_cart_to_motor_side_limit(
            self
    ):
        if not self.motor_side_limit_pressed.is_set():
            self.inside_limits_motor_controller.set_speed(-5)
            self.motor_side_limit_pressed.wait()

        self.outside_limits_motor_controller.set_speed(1)
        self.motor_side_limit_released.wait()

    def motor_side_limit_event(
            self,
            is_pressed: bool
    ):
        if is_pressed:
            self.inside_limits_motor_controller.set_speed(0)
            self.motor_side_limit_pressed.set()
            self.motor_side_limit_released.clear()
        else:
            self.outside_limits_motor_controller.set_speed(0)
            self.degrees_at_motor_side_limit = self.cart_rotary_encoder.degrees
            self.motor_side_limit_released.set()
            self.motor_side_limit_pressed.clear()

    def move_cart_to_rotary_encoder_side_limit(
            self
    ):
        if not self.rotary_encoder_side_limit_pressed.is_set():
            self.inside_limits_motor_controller.set_speed(5)
            self.rotary_encoder_side_limit_pressed.wait()

        self.outside_limits_motor_controller.set_speed(-1)
        self.rotary_encoder_side_limit_released.wait()

    def rotary_encoder_side_limit_event(
            self,
            is_pressed: bool
    ):
        if is_pressed:
            self.inside_limits_motor_controller.set_speed(0)
            self.rotary_encoder_side_limit_pressed.set()
            self.rotary_encoder_side_limit_released.clear()
        else:
            self.outside_limits_motor_controller.set_speed(0)
            self.degrees_at_rotary_encoder_side_limit = self.cart_rotary_encoder.degrees
            self.rotary_encoder_side_limit_released.set()
            self.rotary_encoder_side_limit_pressed.clear()

    def calibrate(
            self
    ):
        self.move_cart_to_motor_side_limit()
        self.move_cart_to_rotary_encoder_side_limit()
        self.limit_to_limit_degrees = abs(self.degrees_at_motor_side_limit - self.degrees_at_rotary_encoder_side_limit)
        self.cart_mm_per_degree = self.limit_to_limit_distance_mm / self.limit_to_limit_degrees

    def center_cart(
            self
    ):
        self.move_cart_to_motor_side_limit()
        self.cart_rotary_encoder.report_state = True
        self.inside_limits_motor_controller.set_speed(5)
        self.cart_rotary_encoder.event(lambda s: (
            self.inside_limits_motor_controller.set_speed(0)
            if (s.degrees - self.degrees_at_motor_side_limit) / self.cart_mm_per_degree >= self.midline_mm
            else None
        ))
        self.cart_rotary_encoder.report_state = False

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

        previous_pole_phase_changes = self.pole_rotary_encoder.phase_changes
        time.sleep(1.0)
        while self.pole_rotary_encoder.phase_changes != previous_pole_phase_changes:
            time.sleep(1.0)

    def advance(
            self,
            state: MdpState,
            t: int,
            a: Action,
            agent: Agent
    ) -> Tuple[MdpState, Reward]:

        pass
