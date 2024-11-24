import argparse
from collections import namedtuple
from typing import Optional

import gymnasium as gym
import numpy as np
import yaml

Task = namedtuple("Task", ["id", "type", "length", "arrival_time"])


class ClusterEnv(gym.Env):
    def __init__(self, args: argparse.Namespace):
        self.device = args.device

        # Set VM configuration
        with open(args.vm_config, "r") as f:
            vm_config = yaml.safe_load(f)
        self.vm_type = np.array(vm_config["type"], dtype=np.int8)
        self.vm_acceleration = np.array(vm_config["acceleration"], dtype=np.float32)
        self.vm_boot_up_cost = np.array(vm_config["boot_up_cost"], dtype=np.float32)
        self.vm_runtime_cost = np.array(vm_config["runtime_cost"], dtype=np.float32)
        if not (
            self.vm_type.shape == self.vm_acceleration.shape == self.vm_boot_up_cost.shape == self.vm_runtime_cost.shape
        ):
            raise ValueError("The shapes of the VM configuration are not consistent.")

        self.vm_num = len(self.vm_type)
        self.base_computing_capacity = args.base_computing_capacity

        # Set task configuration
        self.task_num = args.task_num
        self.io_ratio = args.io_ratio
        self.task_len_mean = args.task_len_mean
        self.task_len_std = args.task_len_std
        self.task_arrival_rate = args.task_arrival_rate
        self.task_timeout = args.task_timeout

        # State of environment
        self.workload: list[Task]
        self.response_time: list[float]  # Response time of each task
        self.is_success: list[bool]  # Whether response time of each task is within timeout
        self.cost: float
        self.vm_idle_time: list[float]
        self.current_task: int

        # Reward function
        self.alpha = args.alpha

        self.action_space = gym.spaces.Discrete(self.vm_num)
        self.observation_space = gym.spaces.Box(
            0.0, np.inf, shape=(self.vm_num + 1,), dtype=np.float32
        )  # Contains current task type and waiting time for each VM

    def __get_observation(self):
        if self.current_task == self.task_num:
            # Return a -1.0 array if all tasks have been processed
            return np.full(self.observation_space.shape, -1.0, dtype=np.float32)
        task_type = self.workload[self.current_task].type
        arrival_time = self.workload[self.current_task].arrival_time
        waiting_time = [max(0, arrival_time - idle_time) for idle_time in self.vm_idle_time]
        return np.array([task_type] + waiting_time, dtype=np.float32)

    def __get_info(self):
        return {}  # TODO: Implement this method

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        task_type = np.random.choice([0, 1], self.task_num, p=[self.io_ratio, 1 - self.io_ratio])
        task_length = np.random.normal(self.task_len_mean, self.task_len_std, self.task_num)
        arrival_interval = np.random.exponential(1.0 / self.task_arrival_rate, self.task_num)
        arrival_time = np.cumsum(arrival_interval)

        self.workload = [Task(i, task_type[i], task_length[i], arrival_time[i]) for i in range(self.task_num)]
        self.response_time = [0.0] * self.task_num
        self.is_success = [False] * self.task_num
        self.cost = 0.0
        self.vm_idle_time = [0.0] * self.vm_num
        self.current_task = 0

        observation = self.__get_observation()
        info = self.__get_info()

        return observation, info

    def step(self, action):
        chosen_vm = action
        task = self.workload[self.current_task]

        # Compute essential information
        idle_time = self.vm_idle_time[chosen_vm]
        start_time = max(idle_time, task.arrival_time)
        execution_time = (
            0.5
            * (1 + int(task.type == self.vm_type[chosen_vm]))
            * task.length
            / (self.base_computing_capacity * self.vm_acceleration[chosen_vm])
        )
        finish_time = start_time + execution_time
        response_time = finish_time - task.arrival_time
        cost = self.vm_boot_up_cost[chosen_vm] + self.vm_runtime_cost[chosen_vm] * execution_time

        # Update environment state
        self.response_time[self.current_task] = response_time
        self.is_success[self.current_task] = response_time <= self.task_timeout
        self.cost += cost
        self.vm_idle_time[chosen_vm] = finish_time
        self.current_task += 1

        observation = self.__get_observation()
        reward = (1 + np.exp(self.alpha - cost)) * execution_time / response_time
        terminated = self.current_task == self.task_num
        truncated = False
        info = self.__get_info()
        # TODO: Add more information to info

        return observation, reward, terminated, truncated, info
