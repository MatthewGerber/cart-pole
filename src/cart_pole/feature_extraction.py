import numpy as np

from cart_pole.environment import CartPoleState
from rlai.models.feature_extraction import StationaryFeatureScaler
from rlai.state_value.function_approximation.models.feature_extraction import (
    StateFeatureExtractor,
    OneHotStateSegmentFeatureInteracter
)


class CartPoleFeatureExtractor(StateFeatureExtractor):
    """
    Feature extractor for the cart-pole environment.
    """

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



        state_feature_vector = np.append([0.01], self.feature_scaler super().extract(state, refit_scaler))
        state_category_feature_vector = self.state_category_interacter.interact(
            np.array([state.observation]),
            np.array([state_feature_vector])
        )[0]

        return state_category_feature_vector

    def __init__(
            self
    ):
        """
        Initialize the feature extractor.
        """

        super().__init__()

        self.feature_scaler = StationaryFeatureScaler()

        # interact features with relevant state categories
        self.state_category_interacter = OneHotStateSegmentFeatureInteracter({

            # cart position
            0: [],

            # cart velocity
            1: [0.0],

            # pole angle
            2: [],

            # pole angular velocity
            3: []
        })
