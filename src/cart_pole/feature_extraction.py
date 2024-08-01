import itertools
import logging
import math
from argparse import ArgumentParser
from typing import List, Tuple, Dict, Optional

import numpy as np

from cart_pole.environment import CartPoleState, CartPole
from rlai.core import MdpState
from rlai.models.feature_extraction import FeatureExtractor, StationaryFeatureScaler
from rlai.state_value.function_approximation.models.feature_extraction import (
    StateFeatureExtractor,
    OneHotStateIndicatorFeatureInteracter,
    StateDimensionLambda,
    StateDimensionSegment
)
from rlai.utils import parse_arguments


class CartPoleBaselineFeatureExtractor(StateFeatureExtractor):
    """
    Feature extractor for the baseline estimator used in the cart-pole environment.
    """

    @classmethod
    def get_argument_parser(
            cls
    ) -> ArgumentParser:
        """
        Get argument parser.

        :return: Argument parser.
        """

        parser = ArgumentParser(
            prog=f'{cls.__module__}.{cls.__name__}',
            parents=[super().get_argument_parser()],
            allow_abbrev=False,
            add_help=False
        )

        return parser

    @classmethod
    def init_from_arguments(
            cls,
            args: List[str],
            environment: CartPole
    ) -> Tuple[FeatureExtractor, List[str]]:
        """
        Initialize a feature extractor from arguments.

        :param args: Arguments.
        :param environment: Environment.
        :return: 2-tuple of a feature extractor and a list of unparsed arguments.
        """

        parsed_args, unparsed_args = parse_arguments(cls, args)

        # there shouldn't be anything left
        if len(vars(parsed_args)) > 0:  # pragma no cover
            raise ValueError('Parsed args remain. Need to pass to constructor.')

        fex = cls(environment)

        return fex, unparsed_args

    def extracts_intercept(
            self
    ) -> bool:
        """
        Whether the feature extractor extracts an intercept (constant) term.

        :return: True if an intercept (constant) term is extracted and False otherwise.
        """

        return True

    def extract(
            self,
            state: CartPoleState,
            refit_scaler: bool
    ) -> np.ndarray:
        """
        Extract state features.

        :param state: State.
        :param refit_scaler: Whether to refit the feature scaler before scaling the extracted features. This is
        only appropriate in settings where nonstationarity is desired (e.g., during training). During evaluation, the
        scaler should remain fixed, which means this should be False.
        :return: State-feature vector.
        """

        # evolve the state forward with constant cart and pole velocities
        evolution_seconds = 1.0
        evolution_steps = math.ceil(evolution_seconds * self.environment.timesteps_per_second)

        step_seconds = [
            step / self.environment.timesteps_per_second
            for step in range(evolution_steps)
        ]

        step_cart_distance_mm = [
            seconds * state.cart_velocity_mm_per_second
            for seconds in step_seconds
        ]

        step_terminal = [
            self.environment.cart_violates_soft_limit(state.cart_mm_from_center + cart_distance_mm)
            for cart_distance_mm in step_cart_distance_mm
        ]

        # only evolve up until first terminal distance
        if step_terminal[-1]:
            num_steps_until_termination = step_terminal.index(True) + 1
            step_seconds = step_seconds[0:num_steps_until_termination]
            step_cart_distance_mm = step_cart_distance_mm[0:num_steps_until_termination]
            step_terminal = step_terminal[0:num_steps_until_termination]

        step_pole_distance_deg = [
            seconds * state.pole_angular_velocity_deg_per_sec
            for seconds in step_seconds
        ]

        step_gamma = [
            (
                1.0 if step == 0
                else self.pre_truncation_gamma if step < self.environment.T
                else self.post_truncation_gamma
            )
            for step in range(len(step_seconds))
        ]

        step_discount = [
            np.prod(step_gamma[0:step + 1])
            for step in range(len(step_gamma))
        ]

        # calculate baseline return as sum of discounted evolved rewards
        baseline_return = np.sum([
            discount * self.environment.get_reward(
                CartPoleState(
                    environment=self.environment,
                    cart_mm_from_center=state.cart_mm_from_center + cart_distance_mm,
                    cart_velocity_mm_per_sec=state.cart_velocity_mm_per_second,
                    pole_angle_deg_from_upright=state.pole_angle_deg_from_upright + pole_distance_deg,
                    pole_angular_velocity_deg_per_sec=state.pole_angular_velocity_deg_per_sec,
                    pole_angular_acceleration_deg_per_sec_squared=state.pole_angular_acceleration_deg_per_sec_squared,
                    step=state.step + step,
                    agent=self.environment.agent,
                    terminal=terminal,
                    truncated=state.truncated
                )
            )
            for step, (discount, cart_distance_mm, pole_distance_deg, terminal) in enumerate(zip(
                step_discount, step_cart_distance_mm, step_pole_distance_deg, step_terminal
            ))
        ])

        return np.array([1.0, baseline_return])

    def reset_for_new_run(
            self,
            state: MdpState
    ):
        """
        Reset for new run.

        :param state: State.
        """

        self.pre_truncation_gamma = self.environment.agent.gamma
        self.post_truncation_gamma = self.environment.truncation_gamma

        logging.info(
            f'Baseline feature extractor reset with pre-truncation gamma={self.pre_truncation_gamma} and '
            f'post-truncation gamma={self.post_truncation_gamma}.'
        )

    def __init__(
            self,
            environment: CartPole
    ):
        """
        Initialize the feature extractor.

        :param environment: Environment.
        """

        super().__init__()

        self.environment = environment
        self.pre_truncation_gamma: Optional[float] = None
        self.post_truncation_gamma: Optional[float] = None


class CartPolePolicyFeatureExtractor(StateFeatureExtractor):
    """
    Feature extractor for the policy used in the cart-pole environment.
    """

    @classmethod
    def get_argument_parser(
            cls
    ) -> ArgumentParser:
        """
        Get argument parser.

        :return: Argument parser.
        """

        parser = ArgumentParser(
            prog=f'{cls.__module__}.{cls.__name__}',
            parents=[super().get_argument_parser()],
            allow_abbrev=False,
            add_help=False
        )

        return parser

    @classmethod
    def init_from_arguments(
            cls,
            args: List[str],
            environment: CartPole
    ) -> Tuple[FeatureExtractor, List[str]]:
        """
        Initialize a feature extractor from arguments.

        :param args: Arguments.
        :param environment: Environment.
        :return: 2-tuple of a feature extractor and a list of unparsed arguments.
        """

        parsed_args, unparsed_args = parse_arguments(cls, args)

        # there shouldn't be anything left
        if len(vars(parsed_args)) > 0:  # pragma no cover
            raise ValueError('Parsed args remain. Need to pass to constructor.')

        fex = cls(environment)

        return fex, unparsed_args

    def extracts_intercept(
            self
    ) -> bool:
        """
        Whether the feature extractor extracts an intercept (constant) term.

        :return: True if an intercept (constant) term is extracted and False otherwise.
        """

        return True

    def extract(
            self,
            state: CartPoleState,
            refit_scaler: bool
    ) -> np.ndarray:
        """
        Extract state features.

        :param state: State.
        :param refit_scaler: Whether to refit the feature scaler before scaling the extracted features. This is
        only appropriate in settings where nonstationarity is desired (e.g., during training). During evaluation, the
        scaler should remain fixed, which means this should be False.
        :return: State-feature vector.
        """

        # obtain the list of term indices that comprise the fully-interacted model. this includes all combinations of
        # terms of each order (e.g., single terms, two-way interactions, three-way, etc.).
        if self.interaction_term_indices is None:
            indices = list(range(len(state.observation)))
            self.interaction_term_indices = [
                list(term_indices_tuple)
                for term_order in range(1, len(state.observation) + 1)
                for term_indices_tuple in itertools.combinations(indices, term_order)
            ]

        # range all features to be in a nominal range of approximately [-1.0, 1.0]. this is only approximate because
        # the minimum and maximum values of some state dimensions (e.g., pole angular velocity) are unlimited in theory
        # and uncalibrated in practice.
        ranged_feature_vector = np.array([

            # cart position:  0.0 in middle, -1.0 at left soft limit, and +1.0 at right soft limit.
            (
                np.sign(state.cart_mm_from_center) *
                (abs(state.cart_mm_from_center) / self.environment.soft_limit_mm_from_midline)
            ),

            # cart velocity:  fraction of maximum speed (calibrated)
            state.cart_velocity_mm_per_second / self.environment.max_cart_speed_mm_per_second,

            # pole angle:  0.0 at bottom, decreasing to -1.0 as pole rises clockwise, and increasing to +1.0 as the
            # pole rises counterclockwise. see the state segmentation for how we partition the policy on this feature.
            state.zero_to_one_pole_angle * np.sign(state.pole_angle_deg_from_upright),

            # pole velocity:  fraction of maximum angular velocity (uncalibrated)
            state.pole_angular_velocity_deg_per_sec / self.environment.max_pole_angular_speed_deg_per_second,

            # pole acceleration:  fraction of maximum angular acceleration (uncalibrated)
            (
                state.pole_angular_acceleration_deg_per_sec_squared /
                self.environment.max_pole_angular_acceleration_deg_per_second_squared
            )
        ])

        # scaling the features to be in [-1.0, 1.0] according to their theoretical bounds doesn't mean that the
        # distribution of observed values will be similar when running. for example, the observed cart and pole
        # velocities might be quite small relative to their ranges and the observed values of cart and pole positions.
        # these differences in observed distribution across features result in the usual issues pertaining to step
        # sizes in the policy updates. standardize the scaled feature vector so that a single step size will be
        # feasible.
        scaled_feature_vector = self.scaler.scale_features(np.array([ranged_feature_vector]), refit_scaler)[0]

        # prepend constant intercept and add multiplicative interaction terms
        state_feature_vector = np.append(
            [1.0],
            [
                np.prod(scaled_feature_vector[term_indices])
                for term_indices in self.interaction_term_indices
            ]
        )

        # interact the feature vector according to its state segment
        state_indicator_feature_vector = self.state_category_interacter.interact(
            np.array([state.observation]),
            np.array([state_feature_vector])
        )[0]

        return state_indicator_feature_vector

    def get_interacter(
            self
    ) -> OneHotStateIndicatorFeatureInteracter:
        """
        Get interacter.

        :return: Interacter.
        """

        return OneHotStateIndicatorFeatureInteracter([

            # segment policy per episode phase
            StateDimensionLambda(
                None,
                lambda _: self.environment.episode_phase.value,
                [episode_phase.value for episode_phase in CartPole.EpisodePhase]
            ),

            # segment policy per pole on either side of vertical. we haven't yet found a feature that quantifies the
            # correct policy response for pole angle. the feature would need to have similar values near either side
            # of vertical downward and similar values near either side of vertical upward, with opposing values
            # depending on whether the pole is on the left half or right half. this segmentation approach splits the
            # policy on left/right half, such that the pole angle feature can reflect the appropriate policy response.
            StateDimensionSegment(
                CartPoleState.Dimension.PoleAngle.value,
                None,
                0.0
            )
        ])

    def __init__(
            self,
            environment: CartPole
    ):
        """
        Initialize the feature extractor.

        :param environment: Environment.
        """

        super().__init__()

        self.environment = environment

        self.scaler = StationaryFeatureScaler()
        self.state_category_interacter = self.get_interacter()
        self.interaction_term_indices: Optional[List[Tuple]] = None

    def __getstate__(
            self
    ) -> Dict:
        """
        Get state for pickling.

        :return: State.
        """

        state = dict(self.__dict__)

        state['state_category_interacter'] = None

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

        state['state_category_interacter'] = self.get_interacter()