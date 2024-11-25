# Cost-Aware

## About `Cost-Aware.py`

[The file](./Cost-Aware.py) `Cost-Aware.py` implements DQN algorithm manually. The only thing you need to do with this file is to understand the DQN algorithm and how it works.

Use the following command to run the file:

```shell
python Cost-Aware.py
```

This repository will use the standardized DRL libraries [Gymnasium](https://gymnasium.farama.org) and [Tianshou](https://tianshou.org/en/stable/) to implement the environment and agent training.

## Python Environment Setup

Create a new conda environment:

```shell
# Create a new conda environment
conda create -n tianshou python=3.11

# Activate the environment
conda activate tianshou
```

Install Tianshou:

```shell
git clone --branch v1.1.0 --depth 1 https://github.com/thu-ml/tianshou.git
cd tianshou
pip install poetry

# Change the source of poetry if necessary
poetry source add --priority=primary tsinghua https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

poetry lock --no-update
poetry install
```

Install other dependencies:

```shell
pip install -r requirements.txt
```

## Getting Started

## Train

Use the following command to train the agent:

```shell
python run.py
```

Use the following command to see adjustable parameters:

```shell
python run.py -h
```

The VM configuration is defined in `./config/vm.yaml`.

The best model will be saved in `./logs/{timestamp}-train/best.pth`.

### Use TensorBoard

```shell
tensorboard --logdir ./logs
```

## Evaluate

Use the following command to evaluate the agent:

```shell
python run.py --eval --model-path ./logs/{timestamp}-train/best.pth
```

`--eval-episode` is used to specify the number of episodes to evaluate the agent.

Use the following command to compare the agent with the baseline and perform significance analysis of the difference:

```shell
python run.py --eval --model-path ./logs/{timestamp}-train/best.pth --baseline --eval-episode 50
```

Use the following command to plot figures:

```shell
python run.py --eval --model-path ./logs/{timestamp}-train/best.pth --baseline --eval-episode 50 --plot
```

## References

<https://github.com/huang1997214/Cost-Aware>

```text
@article{cheng2022cost,
  title={Cost-aware job scheduling for cloud instances using deep reinforcement learning},
  author={Cheng, Feng and Huang, Yifeng and Tanpure, Bhavana and Sawalani, Pawan and Cheng, Long and Liu, Cong},
  journal={Cluster Computing},
  pages={1--13},
  year={2022},
  publisher={Springer}
}
```
