import itertools
from argparse import ArgumentParser
from typing import List, Tuple, Dict, Optional

import numpy as np

from cart_pole.environment import CartPoleState, CartPole, EpisodePhase
from rlai.core import MdpState
from rlai.models.feature_extraction import FeatureExtractor, StationaryFeatureScaler
from rlai.state_value.function_approximation.models.feature_extraction import (
    StateFeatureExtractor,
    OneHotStateIndicatorFeatureInteracter,
    StateDimensionSegment,
    StateLambdaIndicator
)
from rlai.utils import parse_arguments


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
            states: List[MdpState],
            refit_scaler: bool
    ) -> np.ndarray:
        """
        Extract state features.

        :param states: States.
        :param refit_scaler: Whether to refit the feature scaler before scaling the extracted features. This is
        only appropriate in settings where nonstationarity is desired (e.g., during training). During evaluation, the
        scaler should remain fixed, which means this should be False.
        :return: State-feature matrix (#states, #features).
        """

        # obtain the list of term indices that comprise the fully-interacted model. this includes all combinations of
        # terms of each order (e.g., single terms, two-way interactions, three-way, etc.).
        if self.interaction_term_indices is None:
            sample_state = states[0]
            assert isinstance(sample_state, CartPoleState)
            num_state_dims = len(sample_state.observation)
            indices = list(range(num_state_dims))
            self.interaction_term_indices = [
                list(term_indices_tuple)
                for term_order in range(1, num_state_dims + 1)
                for term_indices_tuple in itertools.combinations(indices, term_order)
            ]

        # range all features to be in a nominal range of approximately [-1.0, 1.0]. this is only approximate because
        # the minimum and maximum values of some state dimensions (e.g., pole angular velocity) are unlimited in theory
        # and uncalibrated in practice -- we provide rough estimates manually.
        sign_matrix = np.array([
            [
                np.sign(state.cart_mm_from_center),
                np.sign(state.cart_velocity_mm_per_second),
                np.sign(state.pole_angle_deg_from_upright),
                np.sign(state.pole_angular_velocity_deg_per_sec),
                np.sign(state.pole_angular_acceleration_deg_per_sec_squared)
            ]
            for state in states
            if isinstance(state, CartPoleState)
        ])
        zero_to_one_feature_matrix = np.array([
            [
                state.zero_to_one_cart_distance_from_center,
                state.zero_to_one_cart_speed,
                state.zero_to_one_pole_angle,
                state.zero_to_one_pole_angular_speed,
                state.zero_to_one_pole_angular_acceleration
            ]
            for state in states
            if isinstance(state, CartPoleState)
        ])

        # invert back to 1.0 being most physically extreme
        ranged_feature_matrix = sign_matrix * (1.0 - zero_to_one_feature_matrix)

        # create the full matrix of multiplicative interaction terms between the ranged feature values
        interaction_feature_matrix = np.array([
            [
                np.prod(row[term_indices])
                for term_indices in self.interaction_term_indices
            ]
            for row in ranged_feature_matrix
        ])

        state_indicator_matrix = np.array([
            state.observation
            for state in states
            if isinstance(state, CartPoleState)
        ])

        # interact the feature matrix according to its state segment
        state_indicator_feature_matrix = self.state_category_interacter.interact(
            state_indicator_matrix,
            interaction_feature_matrix
        )

        # ranging the feature values to be in [-1.0, 1.0] according to their theoretical bounds doesn't mean that the
        # distribution of observed values will be similar when running. for example, the observed cart and pole
        # velocities might be quite small relative to their ranges and the observed values of cart and pole positions.
        # these differences in observed distribution across features result in the usual issues pertaining to step
        # sizes in the policy updates. standardize the ranged feature matrix so that a single step size will be
        # suitable for learning. we scale here, after interaction, to ensure that observations in each state segment
        # are scaled according to their naturally observed distributions.
        scaled_state_indicator_feature_matrix = self.scaler.scale_features(
            state_indicator_feature_matrix,
            refit_scaler
        )

        # prepend a vector of intercept terms to each row according to the state segment. we do this here, after
        # scaling, so that the constant intercept terms are not scaled to zero.
        state_indicator_intercepts = self.state_category_interacter.interact(
            state_indicator_matrix,
            0.01 * np.ones((scaled_state_indicator_feature_matrix.shape[0], 1))
        )

        scaled_feature_matrix_with_intercepts = np.append(
            state_indicator_intercepts,
            scaled_state_indicator_feature_matrix,
            axis=1
        )

        return scaled_feature_matrix_with_intercepts

    def get_interacter(
            self
    ) -> OneHotStateIndicatorFeatureInteracter:
        """
        Get interacter.

        :return: Interacter.
        """

        return OneHotStateIndicatorFeatureInteracter([

            # segment policy per pole on either side of vertical. we haven't yet found a feature that quantifies the
            # correct policy response for pole angle. the feature would need to have similar values near either side
            # of vertical downward and similar values near either side of vertical upward, with opposing values
            # depending on whether the pole is on the left half or right half. this segmentation approach splits the
            # policy on left/right half, such that the pole angle feature can reflect the appropriate policy response.
            StateDimensionSegment(
                CartPoleState.Dimension.PoleAngle.value,
                None,
                0.0
            ),

            # segment policy for when the pole is balancing. it is difficult for the swing-up policy to react
            # appropriately in this position, so we use a separate policy for this phase.
            StateLambdaIndicator(
                lambda observation: self.environment.get_episode_phase(
                    observation[CartPoleState.Dimension.PoleAngle.value],
                    observation[CartPoleState.Dimension.PoleVelocity.value]
                ) == EpisodePhase.BALANCE,
                [False, True]
            )
        ])
