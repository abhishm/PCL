# import seaborn
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def return_reward(file_name):
    reward = pickle.load(open(file_name, "rb"))
    return reward

def get_axes():
    grid = GridSpec(2, 4, wspace=1, hspace=0.5)
    ax1 = plt.subplot(grid[0, 1:3])
    ax2 = plt.subplot(grid[1, 0:2])
    ax3 = plt.subplot(grid[1, 2:])
    return (ax1, ax2, ax3)

def plot_all_rewards(file_names, kernel_size=10):
    rewards = map(return_reward, file_names)
    legends = map(lambda s: s + " learning", suffix)
    axes = get_axes()

    for reward, legend, ax in zip(rewards, legends, axes):
        smooth_reward = np.correlate(reward, np.ones(kernel_size) / kernel_size, mode="same")
        ax.plot(smooth_reward[:-kernel_size])
        ax.legend([legend], loc="center right")
        ax.set_xlabel("iteration", fontsize=12)
        ax.set_ylabel("Average return per episode", fontsize=12)
        ax.set_ylim([0, 205])
    plt.suptitle("Different mode of running", fontsize=20)
    fig = plt.gcf()
    fig.set_size_inches(10.5, 7.5)
    fig.savefig("rewards_grid.png", dpi=100)

if __name__ == "__main__":
    suffix = ["online", "offline", "online_offline"]
    file_names = ["rewards_" + s + ".p" for s in suffix]
    plot_all_rewards(file_names)
