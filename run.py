import argparse
import logging

import torch

from env import ClusterEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

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

    if len(parser.parse_known_args()[1]) > 0:
        logger.warning("Unused arguments: %s", parser.parse_known_args()[1])

    return parser.parse_known_args()[0]


if __name__ == "__main__":
    args = get_args()
    logger.info(f"Using device: {args.device}")

    env = ClusterEnv(args)
    env.reset()
