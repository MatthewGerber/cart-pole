import itertools
import math
from argparse import ArgumentParser
from typing import List, Tuple, Dict, Optional

import numpy as np
from cart_pole.environment import CartPoleState, CartPole

from rlai.models.feature_extraction import StationaryFeatureScaler, FeatureExtractor
from rlai.state_value.function_approximation.models.feature_extraction import (
    StateFeatureExtractor,
    OneHotStateIndicatorFeatureInteracter,
    StateDimensionLambda
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
        evolution_seconds = 0.5
        evolution_steps = math.ceil(evolution_seconds * self.environment.timesteps_per_second)

        # noinspection PyUnresolvedReferences
        terminal_values = [
            self.environment.is_terminal(
                state.cart_mm_from_center +
                (step / self.environment.timesteps_per_second) * state.cart_velocity_mm_per_second
            )
            for step in range(evolution_steps)
        ]
        if terminal_values[-1]:
            terminal_values = terminal_values[0:terminal_values.index(True) + 1]

        baseline_return = np.sum([
            self.environment.get_reward(
                CartPoleState(
                    environment=self.environment,
                    cart_mm_from_center=(
                        state.cart_mm_from_center +
                        (step / self.environment.timesteps_per_second) * state.cart_velocity_mm_per_second
                    ),
                    cart_velocity_mm_per_sec=state.cart_velocity_mm_per_second,
                    pole_angle_deg_from_upright=(
                        state.pole_angle_deg_from_upright +
                        (step / self.environment.timesteps_per_second) * state.pole_angular_velocity_deg_per_sec
                    ),
                    pole_angular_velocity_deg_per_sec=state.pole_angular_velocity_deg_per_sec,
                    agent=self.environment.agent,
                    terminal=terminal,
                    truncated=state.truncated
                )
            ) * (self.environment.agent.gamma ** step)
            for step, terminal in enumerate(terminal_values)
        ])

        return np.array([baseline_return])

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

        fex = cls()

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

        if self.term_indices is None:
            indices = list(range(len(state.observation)))
            self.term_indices = [
                list(term_indices_tuple)
                for term_order in range(1, len(state.observation) + 1)
                for term_indices_tuple in itertools.combinations(indices, term_order)
            ]

        scaled_features = self.feature_scaler.scale_features(
            feature_matrix=np.array([
                [
                    state.cart_mm_from_center,
                    state.cart_velocity_mm_per_second,
                    state.pole_angular_velocity_deg_per_sec
                ]
            ]),
            refit_before_scaling=refit_scaler
        )[0]

        scaled_features = np.append(
            scaled_features,
            math.cos(math.pi * (state.pole_angle_deg_from_upright / 180.0))
        )

        state_feature_vector = np.append(
            [1.0],
            [
                np.prod(scaled_features[term_indices])
                for term_indices in self.term_indices
            ]
        )

        state_category_feature_vector = self.state_category_interacter.interact(
            np.array([state.observation]),
            np.array([state_feature_vector])
        )[0]

        return state_category_feature_vector

    @staticmethod
    def get_interacter() -> OneHotStateIndicatorFeatureInteracter:
        """
        Get interacter.

        :return: Interacter.
        """

        return OneHotStateIndicatorFeatureInteracter([

            # use a separate policy when the pole is nearly upright
            StateDimensionLambda(
                2,
                lambda v: (
                    0 if abs(v) <= 30.0
                    else 1
                ),
                list(range(2))
            )

       ])

    def __init__(
            self
    ):
        """
        Initialize the feature extractor.
        """

        super().__init__()

        self.feature_scaler = StationaryFeatureScaler()
        self.term_indices: Optional[List[Tuple]] = None

        # interact features with relevant state categories
        self.state_category_interacter = CartPolePolicyFeatureExtractor.get_interacter()

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

        state['state_category_interacter'] = CartPolePolicyFeatureExtractor.get_interacter()

        self.__dict__ = state
