import numpy as np

from rlai.state_value.function_approximation.models.feature_extraction import OneHotStateIndicatorFeatureInteracter, \
    StateLambdaIndicator


def main():
    """
    Scratch.
    """

    # num_state_dims = 5
    # indices = list(range(num_state_dims))
    # interaction_term_indices = [
    #     list(term_indices_tuple)
    #     for term_order in range(1, num_state_dims + 1)
    #     for term_indices_tuple in itertools.combinations(indices, term_order)
    # ]
    # print(len(interaction_term_indices))

    interacter = OneHotStateIndicatorFeatureInteracter(
        [
            StateLambdaIndicator(lambda v: v[0] < 0.0, [True, False])
        ],
        False
    )

    v = np.array(['a', 'b', 'c'])

    vv = interacter.interact(
        np.array([
            [-1.0]
        ]),
        np.array([
            v
        ]),
        False
    )

    print(f'{vv}')


if __name__ == '__main__':
    main()
