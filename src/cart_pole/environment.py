from typing import List, Tuple

from numpy.random import RandomState

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

    def advance(
            self,
            state: MdpState,
            t: int,
            a: Action,
            agent: Agent
    ) -> Tuple[MdpState, Reward]:
        pass

