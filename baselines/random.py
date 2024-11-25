from typing import Any, Mapping, TypeVar, cast

import gymnasium as gym
import numpy as np

from tianshou.data import Batch
from tianshou.data.batch import BatchProtocol
from tianshou.data.types import ActBatchProtocol, ObsBatchProtocol, RolloutBatchProtocol
from tianshou.policy import BasePolicy
from tianshou.policy.base import TrainingStats


class RandomTrainingStats(TrainingStats):
    pass


TRandomTrainingStats = TypeVar("TRandomTrainingStats", bound=RandomTrainingStats)


class RandomPolicy(BasePolicy[TRandomTrainingStats]):
    def __init__(self, action_space: gym.Space, observation_space: gym.Space | None = None):
        super().__init__(action_space=action_space, observation_space=observation_space)

    def forward(
        self, batch: ObsBatchProtocol, state: dict | BatchProtocol | np.ndarray | None = None, **kwargs: Any
    ) -> ActBatchProtocol:
        result = Batch(act=np.array([self.action_space.sample() for _ in range(len(batch.obs))]))
        return cast(ActBatchProtocol, result)

    def learn(self, batch: RolloutBatchProtocol, *args: Any, **kwargs: Any) -> TRandomTrainingStats:
        """RandomPolicy learns nothing."""
        return RandomTrainingStats()

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        """RandomPolicy does not have any parameters to load."""
        return
