import matplotlib.pyplot as plt

# Global style for more professional look
plt.style.use("seaborn-v0_8-darkgrid")

# Use interactive mode during training, but switch to blocking show at the end.
plt.ion()

PRIMARY_COLOR = "#4e79a7"
SECONDARY_COLOR = "#f28e2c"
ACCENT_COLOR = "#59a14f"
BACKGROUND_COLOR = "#111216"
TEXT_COLOR = "#e0e0e0"
FIG_SIZE = (9, 8)


def _draw_common_layout(fig):
    """Apply common background and spacing."""
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    for ax in fig.get_axes():
        ax.set_facecolor(BACKGROUND_COLOR)
        ax.title.set_color(TEXT_COLOR)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.tick_params(colors=TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_color("#444444")
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_color(TEXT_COLOR)
    plt.tight_layout()


def plot_training(scores, mean_scores, episode_rewards, mses):
    """Robust plotting used during training (can be called many times)."""
    try:
        fig = plt.figure(1, figsize=FIG_SIZE)
        plt.clf()

        # 1) Score progression
        ax1 = plt.subplot(3, 1, 1)
        ax1.set_title("Training progress", fontsize=12, pad=8)
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Score")
        if scores:
            ax1.plot(scores, label="Score per episode", color=PRIMARY_COLOR, linewidth=1.4, alpha=0.9)
        if mean_scores:
            ax1.plot(mean_scores, label="Mean score", color=SECONDARY_COLOR, linewidth=1.6)
        ax1.legend(frameon=False)
        ax1.set_ylim(bottom=0)

        # 2) Reward distribution over episodes
        ax2 = plt.subplot(3, 1, 2)
        ax2.set_title("Reward distribution (total reward per episode)", fontsize=12, pad=8)
        if episode_rewards:
            ax2.hist(episode_rewards, bins=20, color=ACCENT_COLOR, edgecolor="white", alpha=0.8)
        ax2.set_xlabel("Total reward")
        ax2.set_ylabel("Count")

        # 3) MSE (loss) over episodes
        ax3 = plt.subplot(3, 1, 3)
        ax3.set_title("MSE (training loss) over time", fontsize=12, pad=8)
        if mses:
            ax3.plot(mses, label="MSE", color="#e15759", linewidth=1.4)
            ax3.legend(frameon=False)
        ax3.set_xlabel("Episode")
        ax3.set_ylabel("MSE")

        _draw_common_layout(fig)
        plt.draw()
        plt.pause(0.001)
    except Exception as e:
        # Log the error once but do not stop training
        print(f"[Plotting error] {e}")


def show_final_plots(scores, mean_scores, episode_rewards, mses):
    """Blocking version to show final graphs and keep the window open."""
    try:
        plt.ioff()

        fig = plt.figure(1, figsize=FIG_SIZE)
        plt.clf()

        # 1) Score progression
        ax1 = plt.subplot(3, 1, 1)
        ax1.set_title("Training progress", fontsize=12, pad=8)
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Score")
        if scores:
            ax1.plot(scores, label="Score per episode", color=PRIMARY_COLOR, linewidth=1.4, alpha=0.9)
        if mean_scores:
            ax1.plot(mean_scores, label="Mean score", color=SECONDARY_COLOR, linewidth=1.6)
        ax1.legend(frameon=False)
        ax1.set_ylim(bottom=0)

        # 2) Reward distribution over episodes
        ax2 = plt.subplot(3, 1, 2)
        ax2.set_title("Reward distribution (total reward per episode)", fontsize=12, pad=8)
        if episode_rewards:
            ax2.hist(episode_rewards, bins=20, color=ACCENT_COLOR, edgecolor="white", alpha=0.8)
        ax2.set_xlabel("Total reward")
        ax2.set_ylabel("Count")

        # 3) MSE (loss) over episodes
        ax3 = plt.subplot(3, 1, 3)
        ax3.set_title("MSE (training loss) over time", fontsize=12, pad=8)
        if mses:
            ax3.plot(mses, label="MSE", color="#e15759", linewidth=1.4)
            ax3.legend(frameon=False)
        ax3.set_xlabel("Episode")
        ax3.set_ylabel("MSE")

        _draw_common_layout(fig)
        plt.show()
    except Exception as e:
        print(f"[Final plotting error] {e}")

