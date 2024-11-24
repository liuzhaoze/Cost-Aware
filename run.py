import argparse
import os
from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from env import ClusterEnv
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import BasePolicy, DQNPolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", type=str, default="./logs")

    # Parameters for VMs
    parser.add_argument(
        "--vm-config",
        type=str,
        default="./config/vm.yaml",
        help="Path to the VM configuration file.",
    )
    parser.add_argument(
        "--base-computing-capacity",
        type=float,
        default=1000.0,
    )

    # Parameters for tasks
    parser.add_argument("--task-num", type=int, default=500)
    parser.add_argument("--io-ratio", type=float, default=0.5, help="The ratio of I/O tasks.")
    parser.add_argument("--task-len-mean", type=float, default=500.0)
    parser.add_argument("--task-len-std", type=float, default=20.0)
    parser.add_argument("--task-arrival-rate", type=float, default=20.0)
    parser.add_argument(
        "--task-timeout",
        type=float,
        default=0.25,
        help="Task is failed if it's response time exceeds this value.",
    )

    # Parameters for reward function
    parser.add_argument("--alpha", type=float, default=1.5, help="Hyperparameter of reward function.")

    # Parameters for DRL
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[128, 128], help="Hidden layer sizes of DQN.")
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--td-step", type=int, default=3, help="Number of steps for multi-step TD learning.")
    parser.add_argument("--target-update-freq", type=int, default=100, help="Frequency of target network update.")
    parser.add_argument("--buffer-size", type=int, default=100000, help="Size of replay buffer.")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Start value of epsilon-greedy.")
    parser.add_argument("--epsilon-end", type=float, default=0.05, help="End value of epsilon-greedy.")
    parser.add_argument("--epsilon-test", type=float, default=0.01, help="Epsilon for testing.")
    parser.add_argument("--batch-size", type=int, default=64)

    # Parameters for training
    parser.add_argument("--train-env-num", type=int, default=4, help="Number of environments for parallel training.")
    parser.add_argument("--test-env-num", type=int, default=2, help="Number of environments for testing the policy.")
    parser.add_argument("--reward-threshold", type=float, default=1e10, help="Threshold of reward for early stopping.")
    parser.add_argument(
        "--epoch-num",
        type=int,
        default=1500,
        help="Number of epochs for training agent.\nAt the end of each epoch, the policy is evaluated on the test environments. If a new maximum reward is achieved, save the policy.",
    )
    parser.add_argument("--step-per-epoch", type=int, default=2000, help="Number of transitions collected per epoch.")
    parser.add_argument(
        "--step-per-collect",
        type=int,
        default=100,
        help="Number of transitions the collector would collect before the network update",
    )
    parser.add_argument(
        "--update-per-step",
        type=float,
        default=0.1,
        help="How many gradient steps to perform per step in the environment.\n1.0 means perform one gradient step per environment step.\n0.1 means perform one gradient step per 10 environment steps.",
    )
    parser.add_argument("--episode-per-test", type=int, default=4, help="Number of episodes for one policy evaluation.")

    # Parameters for evaluation
    parser.add_argument("--eval", action="store_true", help="Evaluate the policy.")
    parser.add_argument("--model-path", type=str, default=None, help="Path to the model for evaluation.")
    parser.add_argument("--eval-episode", type=int, default=1, help="Number of episodes for evaluation.")

    if len(parser.parse_known_args()[1]) > 0:
        print("Unknown arguments:", parser.parse_known_args()[1])

    return parser.parse_known_args()[0]


def set_log_path(args: argparse.Namespace) -> argparse.Namespace:
    args.now = datetime.now().strftime("%Y%m%d-%H%M%S")
    args.mode = "eval" if args.eval else "train"
    args.log_path = os.path.join(args.log_dir, f"{args.now}-{args.mode}")
    return args


def get_env_info(args: argparse.Namespace) -> argparse.Namespace:
    env = ClusterEnv(args)
    args.state_space = env.observation_space
    args.state_shape = args.state_space.shape or int(args.state_space.n)
    args.action_space = env.action_space
    args.action_shape = args.action_space.shape or int(args.action_space.n)
    return args


def get_policy(
    args: argparse.Namespace, policy: BasePolicy | None = None, optimizer: torch.optim.Optimizer | None = None
):
    if policy is None:
        net = Net(
            state_shape=args.state_shape,
            action_shape=args.action_shape,
            hidden_sizes=args.hidden_sizes,
            device=args.device,
        ).to(args.device)

        if optimizer is None:
            optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)

        policy = DQNPolicy(
            model=net,
            optim=optimizer,
            action_space=args.action_space,
            discount_factor=args.gamma,
            estimation_step=args.td_step,
            target_update_freq=args.target_update_freq,
        )

    if args.eval:
        if args.model_path is None:
            raise ValueError("Model path is required for evaluation. Train a model first.")
        policy.load_state_dict(torch.load(args.model_path))

    return policy


def train(args: argparse.Namespace, policy: BasePolicy | None = None, optimizer: torch.optim.Optimizer | None = None):
    train_envs = DummyVectorEnv([lambda: ClusterEnv(args) for _ in range(args.train_env_num)])
    test_envs = DummyVectorEnv([lambda: ClusterEnv(args) for _ in range(args.test_env_num)])

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    # Initialize policy
    policy = get_policy(args, policy, optimizer)

    # Replay buffer
    buffer = VectorReplayBuffer(total_size=args.buffer_size, buffer_num=args.train_env_num)

    # Collector
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)

    # Logger
    writer = SummaryWriter(args.log_path)
    logger = TensorboardLogger(writer)

    # Train
    def train_fn(epoch_num: int, step_idx: int) -> None:
        # Epsilon delay in the first 1M steps
        if step_idx < 1e6:
            epsilon = args.epsilon_end + (args.epsilon_start - args.epsilon_end) * np.exp(-1.0 * step_idx / 1e6)
        else:
            epsilon = args.epsilon_end
        policy.set_eps(epsilon)

    def test_fn(epoch_num: int, step_idx: int) -> None:
        policy.set_eps(args.epsilon_test)

    def stop_fn(mean_rewards: float) -> bool:
        return mean_rewards >= args.reward_threshold

    def save_best_fn(policy: BasePolicy) -> None:
        path = os.path.join(args.log_path, "best.pth")
        torch.save(policy.state_dict(), path)
        print(f"Save the best policy to {path}")

    result = OffpolicyTrainer(
        policy=policy,
        max_epoch=args.epoch_num,
        batch_size=args.batch_size,
        train_collector=train_collector,
        test_collector=test_collector,
        step_per_epoch=args.step_per_epoch,
        step_per_collect=args.step_per_collect,
        update_per_step=args.update_per_step,
        episode_per_test=args.episode_per_test,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
    ).run()

    return result, policy


def evaluate(args: argparse.Namespace, policy: BasePolicy | None = None):
    eval_envs = DummyVectorEnv([lambda: ClusterEnv(args) for _ in range(args.eval_episode)])

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    eval_envs.seed(args.seed)

    # Initialize policy
    policy = get_policy(args, policy)
    policy.eval()
    policy.set_eps(args.epsilon_test)

    # Replay buffer
    buffer = VectorReplayBuffer(total_size=(args.task_num + 10) * args.eval_episode, buffer_num=args.eval_episode)

    # Collector
    collector = Collector(policy, eval_envs, buffer, exploration_noise=True)

    # Evaluate
    result = collector.collect(n_episode=args.eval_episode, reset_before_collect=True)
    metrics = {
        "average_response_time": [
            buffer.buffers[i].info.average_response_time.max() for i in range(len(buffer.buffers))
        ],
        "success_rate": [buffer.buffers[i].info.success_rate.max() for i in range(len(buffer.buffers))],
        "cost": [buffer.buffers[i].info.cost.max() for i in range(len(buffer.buffers))],
        "is_from_same_env": [
            np.all(buffer.buffers[i].info.env_id[: len(buffer.buffers[i])] == buffer.buffers[i].info.env_id[0])
            for i in range(len(buffer.buffers))
        ],
    }
    return result, metrics


if __name__ == "__main__":
    args = get_args()
    args = set_log_path(args)
    args = get_env_info(args)
    print(
        f"Mode: {args.mode} | Device: {args.device}\nState space: {args.state_space} | State shape: {args.state_shape} | Action space: {args.action_space} | Action shape: {args.action_shape}"
    )

    if not args.eval:
        result, policy = train(args)
        print(result)
    else:
        result, metrics = evaluate(args)
        print(result, metrics, sep="\n")
