import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

__all__ = ["statistical_analysis", "draw_chart"]


def statistical_analysis(data: dict, target: str, baselines: list):
    """Use a t-test to compare the significant differences between different policies."""

    metrics = list(data[target].keys())
    is_significant_diff = {}

    for metric in metrics:
        is_significant_diff[metric] = []
        for baseline in baselines:
            t, p = ttest_ind(data[baseline][metric], data[target][metric])
            is_significant_diff[metric].append(p < 0.05)
    print("\nSignificant differences between DQN and other baselines:")
    print(pd.DataFrame(is_significant_diff, index=baselines))


def draw_chart(args: argparse.Namespace, data: dict, policies: list):
    figures = set()
    for p in policies:
        for metric, values in data[p].items():
            figures.add(metric)
            plt.figure(metric)
            plt.bar(p, np.mean(values))
            plt.errorbar(p, np.mean(values), np.std(values), ecolor="k", elinewidth=1.5, capsize=10, capthick=1.5)
            plt.ylabel(metric)
    for f in figures:
        plt.figure(f)
        path = os.path.join(args.fig_dir, f + ".pdf")
        plt.savefig(path)
        print(f"Figure saved to {path}")
    plt.show()
