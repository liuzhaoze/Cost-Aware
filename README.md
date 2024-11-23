# Cost-Aware

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
