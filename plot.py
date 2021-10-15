import os

import numpy as np
import matplotlib.pyplot as plt

possible_algorithms = ['CycleGAN']
colors = ['C0']

def plot_figure(dirname, fname, possible_algorithms=possible_algorithms):
    for algo, c in zip(possible_algorithms, colors):
        try:
            data = np.loadtxt(os.path.join(dirname, "stats", fname), delimiter=',')
            x, y = data[:, 0], data[:, 1]
            plt.title(fname)
            plt.plot(x, y, label=algo, color=c)
        except:
            pass
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(dirname, fname.split(".")[0]))
    plt.close()

if __name__ == '__main__':
    dirpaths = [
        "exp_music/JC_J_JC_C_2021_10_13_00_18_49"
    ]

    for dpth in dirpaths:
        try:
            plot_figure(dpth, "train_stats_d_loss.csv")
            plot_figure(dpth, "train_stats_g_loss.csv")
            plot_figure(dpth, "test_stats_d_loss.csv")
            plot_figure(dpth, "test_stats_g_loss.csv")
        except Exception as e:
            print(e, 'skip', dpth)
