import math
import random
from collections import namedtuple
from itertools import count

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DQN(nn.Module):
    """
    PyTorch 神经网络输入输出为行向量\n
    当矩阵输入神经网络时，神经网络将逐行输入矩阵中的行向量\n
    输出矩阵的每一行是输入行向量的结果
    """

    def __init__(self, num_state_features: int, num_actions: int) -> None:
        super().__init__()

        self.fc1 = nn.Linear(in_features=num_state_features, out_features=20)
        self.out = nn.Linear(in_features=20, out_features=num_actions)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = F.relu(self.fc1(t))
        t = self.out(t)
        return t


Experience = namedtuple("Experience", ["state", "action", "reward", "next_state"])


class ReplayMemory:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

    def push(self, experience: Experience) -> None:
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience

        self.push_count += 1

    def sample(self, batch_size: int) -> list:
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size: int) -> bool:
        return len(self.memory) >= batch_size


class EpsilonGreedy:
    def __init__(self, start: float, end: float, decay: float) -> None:
        self.start = start
        self.end = end
        self.decay = decay

    def get_epsilon(self, current_step: int) -> float:
        return self.end + (self.start - self.end) * math.exp(
            -1.0 * current_step * self.decay
        )


class DRLAgent:
    def __init__(
        self, strategy: EpsilonGreedy, num_actions: int, device: torch.device
    ) -> None:
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device

    def select_action(self, policy_network: DQN, state: torch.Tensor) -> torch.Tensor:
        epsilon = self.strategy.get_epsilon(self.current_step)
        self.current_step += 1

        if random.random() < epsilon:
            # explore
            action = random.randrange(self.num_actions)
            return torch.tensor([action], device=self.device)
        else:
            # exploit
            with torch.no_grad():  # 使用模型推理时关闭梯度跟踪
                """
                unsqueeze: 在张量的最前面添加一个维度，变成行向量
                argmax: 在列的维度获得最大值的索引
                """
                return (
                    policy_network(state).unsqueeze(dim=0).argmax(dim=1).to(self.device)
                )
                # 推理结束后自动打开梯度跟踪


Job = namedtuple("Job", ["id", "type", "submit_time", "length"])


class JobVirtualMachineEnv:
    def __init__(self, device: torch.device) -> None:
        self.device = device

        self.computing_capacity = 1000.0
        self.type_VMs = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])  # 0 - IO; 1 - Compute
        self.boot_up_cost_VMs = np.array(
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        )
        self.runtime_cost_VMs = np.array([1, 1, 2, 2, 4, 1, 1, 2, 2, 4])
        self.acceleration_VMs = np.array([1, 1, 1.1, 1.1, 1.2, 1, 1, 1.1, 1.1, 1.2])
        self.idle_time_VMs = np.zeros(self.type_VMs.shape)

        self.num_jobs = 500
        self.job_type_ratio = 0.5  # IO / total
        self.job_len_mean = 500.0
        self.job_len_std = 20.0
        self.job_lambda = 20
        self.workload = []  # use reset() to generate workload
        self.jobs_response_time = []  # store response time of each job

        self.total_cost = 0.0
        self.current_job = None
        self.done = False

    def num_actions(self) -> int:
        return len(self.type_VMs)

    def num_state_features(self) -> int:
        return len(self.type_VMs) + 1  # wait time of all VMs and current job type

    def reset(self) -> None:
        self.idle_time_VMs = np.zeros(self.type_VMs.shape)
        self.workload = []
        self.jobs_response_time = []
        self.total_cost = 0.0
        self.current_job = 0
        self.done = False

        submit_interval = np.random.exponential(
            1.0 / self.job_lambda, self.num_jobs
        )  # 任务的提交是泊松过程 => 任务提交的间隔服从指数分布
        jobs_length = np.random.normal(
            self.job_len_mean, self.job_len_std, self.num_jobs
        )
        submit_time = 0
        for id in range(self.num_jobs):
            submit_time += submit_interval[id]
            self.workload.append(
                Job(
                    id,
                    0 if np.random.random() < self.job_type_ratio else 1,
                    submit_time,
                    jobs_length[id],
                )
            )

    def get_state(self) -> torch.Tensor:
        if self.done:
            # 调度结束时返回全 -1 向量
            # 不能用全 0 向量，因为如果第一个任务类型为 0 ，所有虚拟机的 idle time 都为 0 ，这种情况不是调度结束的情况
            return (
                torch.zeros(self.num_state_features(), device=self.device).float() - 1
            )

        job_type = self.workload[self.current_job].type
        submit_time = self.workload[self.current_job].submit_time
        wait_time = (self.idle_time_VMs - submit_time).clip(
            min=0
        )  # 空闲时刻减提交时刻小于零说明等待时间为0
        return torch.tensor(
            np.concatenate(([job_type], wait_time)),
            device=self.device,
        ).float()

    def take_action(self, action: torch.Tensor) -> torch.Tensor:
        chosen_VM = action.item()
        idle_time = self.idle_time_VMs[chosen_VM]

        job = self.workload[self.current_job]
        execution_time = (
            0.5
            * ((job.type ^ self.type_VMs[chosen_VM]) + 1)
            * job.length
            / (self.computing_capacity * self.acceleration_VMs[chosen_VM])
        )
        start_time = job.submit_time if job.submit_time >= idle_time else idle_time
        end_time = start_time + execution_time
        wait_time = max(idle_time - job.submit_time, 0)

        cost = (
            self.boot_up_cost_VMs[chosen_VM]
            + execution_time * self.runtime_cost_VMs[chosen_VM]
        )
        reward = (
            (1 + math.exp(1.5 - cost)) * (execution_time) / (execution_time + wait_time)
        )

        self.jobs_response_time.append(wait_time + execution_time)
        self.total_cost += cost
        self.idle_time_VMs[chosen_VM] = end_time
        self.current_job += 1
        self.done = self.current_job == self.num_jobs

        return torch.tensor([reward], device=self.device).float()


class QValues:
    """提供计算当前 Q 值和下一个 Q 值的方法"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def get_current(
        policy_net: DQN, states: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """
        dim 0: 每个样本状态的 Q 值\n
        dim 1: 所有 action 的 Q 值\n
        gather: 选出 actions 对应的 Q 值\n
        unsqueese: 给张量的指定位置添加一个维度
        """
        return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))

    @staticmethod
    def get_next(target_net: DQN, next_states: torch.Tensor) -> torch.Tensor:
        """
        max: 取所有 action 的最大 Q 值
        [0]: max 返回类型的第 0 个元素是最大值
        [1]: max 返回类型的第 1 个元素是最大值对应的索引
        """
        # 找到表示终止状态的全 -1 向量的索引
        final_state_locations = (
            next_states.flatten(start_dim=1).max(dim=1)[0].eq(-1).type(torch.bool)
        )
        # 非终止状态的索引
        non_final_state_locations = final_state_locations == False
        # 非终止状态的样本
        non_final_states = next_states[non_final_state_locations]

        next_q_values = torch.zeros(next_states.shape[0], device=QValues.device)
        next_q_values[non_final_state_locations] = (
            target_net(non_final_states).max(dim=1)[0].detach()
        )
        """
        计算 TD target: target_q_value = reward + (next_q_value * gamma) 时
        如果 next_state 是终止状态， reward 已经是 return 中的最后一项，不需要加上 gamma * next_q_value
        因此终止状态的 next_q_value 设置为 0 ，非终止状态的 next_q_value 从 target_net 中获取
        """
        return next_q_values


def extract_tensors(experiences: list):
    batch = Experience(*zip(*experiences))
    return (
        torch.stack(batch.state),
        torch.cat(batch.action),
        torch.cat(batch.reward),
        torch.stack(batch.next_state),
    )


def plot(**data_dict):
    n_rows = 1
    n_cols = int(len(data_dict) / n_rows)

    plt.figure(1, figsize=(12, 5))
    plt.clf()

    for i, (name, values) in enumerate(data_dict.items()):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.xlabel("Episode")
        plt.ylabel(name)
        plt.plot(values)

    plt.pause(0.01)


"""baseline"""


class RandomAgent:
    def __init__(self, num_actions: int, device: torch.device) -> None:
        self.num_actions = num_actions
        self.device = device

    def select_action(self) -> torch.Tensor:
        action = random.randrange(self.num_actions)
        return torch.tensor([action], device=self.device)


class RoundRobinAgent:
    def __init__(self, num_actions: int, device: torch.device) -> None:
        self.num_actions = num_actions
        self.device = device
        self.current_action = -1

    def select_action(self) -> torch.Tensor:
        self.current_action = (self.current_action + 1) % self.num_actions
        return torch.tensor([self.current_action], device=self.device)


class EarliestAgent:
    def __init__(self, device: torch.device) -> None:
        self.device = device

    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        wait_times = state[1:]
        return wait_times.argmin().unsqueeze(-1).to(self.device)


if __name__ == "__main__":
    batch_size = 256
    gamma = 0.999
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.001
    target_network_update_step = 10
    memory_size = 100000
    learning_rate = 0.001
    num_episodes = 40
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = JobVirtualMachineEnv(device)
    epsilon_greedy = EpsilonGreedy(epsilon_start, epsilon_end, epsilon_decay)
    drl_agent = DRLAgent(epsilon_greedy, env.num_actions(), device)
    memory = ReplayMemory(memory_size)
    policy_net = DQN(env.num_state_features(), env.num_actions()).to(device)
    target_net = DQN(env.num_state_features(), env.num_actions()).to(device)
    target_net.load_state_dict(policy_net.state_dict())  # 拷贝参数
    target_net.eval()  # target_net 只推理不训练
    optimizer = optim.Adam(params=policy_net.parameters(), lr=learning_rate)

    average_response_time = []
    total_costs = []
    for episode in range(num_episodes):
        env.reset()
        state = env.get_state()

        for step in count():
            action = drl_agent.select_action(policy_net, state)
            reward = env.take_action(action)
            next_state = env.get_state()
            memory.push(Experience(state, action, reward, next_state))
            state = next_state

            if memory.can_provide_sample(batch_size):
                experiences = memory.sample(batch_size)
                states, actions, rewards, next_states = extract_tensors(experiences)

                current_q_values = QValues.get_current(policy_net, states, actions)
                next_q_values = QValues.get_next(target_net, next_states)
                target_q_values = rewards + (gamma * next_q_values)

                loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
                optimizer.zero_grad()  # 清空累积梯度
                loss.backward()
                optimizer.step()

            if env.done:
                average_response_time.append(np.mean(env.jobs_response_time))
                total_costs.append(env.total_cost)
                plot(
                    average_response_time=average_response_time,
                    total_cost=total_costs,
                )
                break

        if episode % target_network_update_step == 0:
            target_net.load_state_dict(policy_net.state_dict())

    print("Average Response Time:", average_response_time)
    print("Total Cost:", total_costs)

    # 与 baseline 对比
    import copy

    random_agent = RandomAgent(env.num_actions(), device)
    round_robin_agent = RoundRobinAgent(env.num_actions(), device)
    earliest_agent = EarliestAgent(device)
    agent_names = ["drl", "random", "round_robin", "earliest"]
    agents = [drl_agent, random_agent, round_robin_agent, earliest_agent]
    policy_net.eval()  # 关闭梯度跟踪，只推理不训练
    # 评价指标
    response_time = {}
    cost = {}

    env.reset()
    for name, agent in zip(agent_names, agents):
        # 开始评估
        env_eval = copy.deepcopy(env)  # 所有评估使用同一个环境
        state = env_eval.get_state()

        for step in count():
            match name:
                case "drl":
                    action = agent.select_action(policy_net, state)
                case "random":
                    action = agent.select_action()
                case "round_robin":
                    action = agent.select_action()
                case "earliest":
                    action = agent.select_action(state)
            reward = env_eval.take_action(action)
            next_state = env_eval.get_state()
            state = next_state

            if env_eval.done:
                response_time[name] = np.mean(env_eval.jobs_response_time)
                cost[name] = env_eval.total_cost
                break

    plt.figure(2, figsize=(12, 5))
    plt.clf()
    plt.subplot(121)
    plt.title("Average Response Time")
    for name, data in response_time.items():
        plt.bar(name, data)
    plt.subplot(122)
    plt.title("Total Cost")
    for name, data in cost.items():
        plt.bar(name, data)

    print("Average Response Time:", response_time)
    print("Total Cost:", cost)
    plt.show()
