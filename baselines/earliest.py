from typing import Any, Mapping, TypeVar, cast

import gymnasium as gym
import numpy as np

from tianshou.data import Batch
from tianshou.data.batch import BatchProtocol
from tianshou.data.types import ActBatchProtocol, ObsBatchProtocol, RolloutBatchProtocol
from tianshou.policy import BasePolicy
from tianshou.policy.base import TrainingStats


class EarliestTrainingStats(TrainingStats):
    pass


TEarliestTrainingStats = TypeVar("TEarliestTrainingStats", bound=EarliestTrainingStats)


class EarliestPolicy(BasePolicy[TEarliestTrainingStats]):
    def __init__(self, action_space: gym.Space, observation_space: gym.Space | None = None):
        super().__init__(action_space=action_space, observation_space=observation_space)

    def forward(
        self, batch: ObsBatchProtocol, state: dict | BatchProtocol | None = None, **kwargs: Any
    ) -> ActBatchProtocol:
        waiting_time = batch.obs[:, 1:]
        result = Batch(act=np.argmin(waiting_time, axis=1))
        return cast(ActBatchProtocol, result)

    def learn(self, batch: RolloutBatchProtocol, *args: Any, **kwargs: Any) -> TEarliestTrainingStats:
        """EarliestPolicy learns nothing."""
        return EarliestTrainingStats()

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        """EarliestPolicy does not have any parameters to load."""
        return
