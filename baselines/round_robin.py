from typing import Any, Mapping, TypeVar, cast

import gymnasium as gym
import numpy as np

from tianshou.data import Batch
from tianshou.data.batch import BatchProtocol
from tianshou.data.types import ActBatchProtocol, ObsBatchProtocol, RolloutBatchProtocol
from tianshou.policy import BasePolicy
from tianshou.policy.base import TrainingStats


class RoundRobinTrainingStats(TrainingStats):
    pass


TRoundRobinTrainingStats = TypeVar("TRoundRobinTrainingStats", bound=RoundRobinTrainingStats)


class RoundRobinPolicy(BasePolicy[TRoundRobinTrainingStats]):
    def __init__(self, action_space: gym.Space, observation_space: gym.Space | None = None):
        super().__init__(action_space=action_space, observation_space=observation_space)
        self.last_action = -1

    def forward(
        self, batch: ObsBatchProtocol, state: dict | BatchProtocol | np.ndarray | None = None, **kwargs: Any
    ) -> ActBatchProtocol:
        action_range = self.action_space.shape or int(self.action_space.n)
        self.last_action = (self.last_action + 1) % action_range
        result = Batch(act=np.array([self.last_action for _ in range(len(batch.obs))]))
        return cast(ActBatchProtocol, result)

    def learn(self, batch: RolloutBatchProtocol, *args: Any, **kwargs: Any) -> TRoundRobinTrainingStats:
        """RoundRobinPolicy learns nothing."""
        return RoundRobinTrainingStats()

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        """RoundRobinPolicy does not have any parameters to load."""
        return
