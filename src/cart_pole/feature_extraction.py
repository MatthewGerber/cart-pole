from argparse import ArgumentParser
from typing import List, Tuple, Dict

import numpy as np
from cart_pole.environment import CartPoleState, CartPole

from rlai.models.feature_extraction import StationaryFeatureScaler, FeatureExtractor
from rlai.state_value.function_approximation.models.feature_extraction import (
    StateFeatureExtractor,
    OneHotStateIndicatorFeatureInteracter,
    StateDimensionLambda
)
from rlai.utils import parse_arguments


class CartPoleFeatureExtractor(StateFeatureExtractor):
    """
    Feature extractor for the cart-pole environment.
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

        state_feature_vector = np.append(
            [0.01],
            self.feature_scaler.scale_features(
                np.array([state.observation]),
                refit_before_scaling=refit_scaler
            )[0]
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

            # pole angle
            StateDimensionLambda(
                2,
                lambda v: (
                    0 if abs(v) <= 30.0
                    else 1 if abs(v) <= 90.0
                    else 2
                ),
                list(range(3))
            ),

            # pole angular velocity
            StateDimensionLambda(
                3,
                lambda v: (
                    0 if abs(v) <= 90.0
                    else 1 if abs(v) <= 180.0
                    else 2 if abs(v) <= 360.0
                    else 3
                ),
                list(range(4))
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

        # interact features with relevant state categories
        self.state_category_interacter = CartPoleFeatureExtractor.get_interacter()

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

        state['state_category_interacter'] = CartPoleFeatureExtractor.get_interacter()

        self.__dict__ = state
