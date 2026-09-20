import matplotlib.pyplot as plt
import math
from pathlib import Path


def plotting_logs(
    train_logs: dict[str, list],
    val_logs: dict[str, list],
    path: Path,
    cols_mult: float = 6.5,
    rows_mult: float = 5.5,
):

    key_logs = set(list(train_logs.keys()) + list(val_logs.keys()))

    num_plots = len(key_logs)
    if num_plots == 0:
        print("No logs provided to plot.")
        return

    cols = math.ceil(math.sqrt(num_plots))
    rows = math.ceil(num_plots / cols)

    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * cols_mult, rows * rows_mult), squeeze=False
    )
    axes = axes.flatten()

    for idx, metric_name in enumerate(key_logs):
        ax = axes[idx]

        if metric_name in train_logs:
            ax.plot(
                train_logs[metric_name],
                marker="o",
                linestyle="-",
                linewidth=1.5,
                markersize=3,
                label="train",
            )
        if metric_name in val_logs:
            ax.plot(
                val_logs[metric_name],
                marker="o",
                linestyle="-",
                linewidth=1.5,
                markersize=3,
                label="validation",
            )

        ax.set_title(metric_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("EPOCHS")
        ax.set_ylabel("Value")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()

    for idx in range(num_plots, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.savefig(path)
    plt.clf()
