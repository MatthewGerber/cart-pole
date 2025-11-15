import itertools
import math
from argparse import ArgumentParser
from copy import deepcopy
from typing import List, Tuple, Dict, cast, Optional

import numpy as np
from cart_pole.environment import CartPoleState, CartPole, EpisodePhase
from rlai.core import MdpState
from rlai.models.feature_extraction import FeatureExtractor
from rlai.state_value.function_approximation.models.feature_extraction import (
    StateFeatureExtractor,
    OneHotStateIndicatorFeatureInteracter,
    StateLambdaIndicator,
    StateIndicator
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

        super().__init__(True)

        self.environment = environment

        self.interaction_term_indices: Optional[List[Tuple]] = None

        indicators = self.get_state_indicators()

        # it's important to scale features independently within each state segment, since we'll make many observations
        # in certain segments early in the learning before ever getting to the other segments (e.g., for balancing). we
        # don't want the early zero-valued features for unobserved segments (due to one-hot segment encoding) to pollute
        # the scalers used for segments observed later in learning.
        self.state_category_feature_interacter = OneHotStateIndicatorFeatureInteracter(indicators, False)

        # do not scale intercepts when encoding them
        self.state_category_intercept_interacter = OneHotStateIndicatorFeatureInteracter(indicators, False)

    def __getstate__(
            self
    ) -> Dict:
        """
        Get state for pickling.

        :return: State.
        """

        state = dict(self.__dict__)

        # can't pickle/deepcopy the indicators, one of which has a lambda function in it.
        state['state_category_feature_interacter'].indicators = None
        state['state_category_feature_interacter'] = deepcopy(state['state_category_feature_interacter'])

        state['state_category_intercept_interacter'].indicators = None
        state['state_category_intercept_interacter'] = deepcopy(state['state_category_intercept_interacter'])

        # restore the indicators
        indicators = self.get_state_indicators()
        self.state_category_feature_interacter.indicators = indicators
        self.state_category_intercept_interacter.indicators = indicators

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

        # restore the indicators
        indicators = self.get_state_indicators()
        self.state_category_feature_interacter.indicators = indicators
        self.state_category_intercept_interacter.indicators = indicators

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
            num_state_dims = 4
            term_indices = list(range(num_state_dims))
            self.interaction_term_indices = [
                list(term_indices_tuple)
                for term_order in range(1, num_state_dims + 1)
                for term_indices_tuple in itertools.combinations(term_indices, term_order)
            ]

        # get the raw state matrix, with one row per observation. this is used for feature-space segmentation. scale
        # the cart pos/vel (pos ranges [-400,400]) to be similar in magnitude to pole radians (-pi,+pi).
        state_matrix = np.array([
            cast(CartPoleState, state).observation[[
                CartPoleState.Dimension.CartPosition,
                CartPoleState.Dimension.CartVelocity,
                CartPoleState.Dimension.PoleAngle,
                CartPoleState.Dimension.PoleVelocity,
            ]] * [0.01, 0.01, math.pi / 180.0, math.pi / 180.0]
            for state in states
        ])

        # create the full feature matrix of multiplicative interaction terms
        state_feature_matrix = np.array([
            [
                np.prod(state_row[term_indices])
                for term_indices in self.interaction_term_indices
            ]
            for state_row in state_matrix
        ])

        # interact the feature matrix according to its state segment. this will scale features if enabled on the
        # interacter.
        scaled_state_indicator_feature_matrix = self.state_category_feature_interacter.interact(
            state_matrix,
            state_feature_matrix,
            refit_scaler
        )

        # obtain a vector of intercept terms for each row according to the state segment. we do this here, after
        # scaling, so that the constant intercept terms are not scaled to zero.
        state_indicator_intercepts = self.state_category_intercept_interacter.interact(
            state_matrix,
            np.ones((scaled_state_indicator_feature_matrix.shape[0], 1)),
            False
        )

        # combine intercept columns with feature columns
        scaled_state_indicator_feature_matrix_with_intercepts = np.append(
            state_indicator_intercepts,
            scaled_state_indicator_feature_matrix,
            axis=1
        )

        return scaled_state_indicator_feature_matrix_with_intercepts

    def get_state_indicators(
            self
    ) -> List[StateIndicator]:
        """
        Get state indicators for feature interacters.

        :return: Indicators.
        """

        indicators = [

            # segment policy for when the pole is balancing. it is difficult for the swing-up policy to react
            # appropriately in this position, so we use a separate policy for this phase.
            StateLambdaIndicator(
                lambda feature_vector: self.environment.get_episode_phase(
                    math.degrees(feature_vector[2]),
                    math.degrees(feature_vector[3])
                ) == EpisodePhase.BALANCE,
                [False, True]
            )
        ]

        return indicators
