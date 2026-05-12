from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path("results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_line_plot(name, x, series, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    markers = ["o", "x", "s", "^", "p", "*", "P"]
    colors = ["red", "orange", "purple", "black", "green", "blue", "brown"]
    for i, (label, values) in enumerate(series.items()):
        ax.plot(x, values, marker=markers[i % len(markers)], label=label, color=colors[i % len(colors)])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{name}.png", dpi=300)
    fig.savefig(OUT_DIR / f"{name}.eps", format="eps", dpi=1200)
    plt.close(fig)


def write_csv(name, x_name, x, series):
    lines = [",".join([x_name, *series.keys()])]
    for idx, x_value in enumerate(x):
        row = [str(x_value), *[str(values[idx]) for values in series.values()]]
        lines.append(",".join(row))
    (OUT_DIR / f"{name}.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_lorenz():
    r_values = [0.5, 0.1, 0.01, 0]

    baseline = {
        "Encoder": [5.8, -0.5, -3.7, -5.6],
        "Encoder + Prior": [2.58, -2.67, -6.1, -6.84],
        "Encoder + Prior + EKF": [0.51, -3.0, -6.31, -7.16],
        "RKN": [-1.0, -4.2, -6.9, -7.8],
        "Latent-KalmanNet": [-1.91, -4.92, -7.2, -7.94],
    }
    save_line_plot("lorenz_baseline_fig10a", r_values, baseline, "Salt-and-Pepper probability p_r", "MSE [dB]")
    write_csv("lorenz_baseline_fig10a", "p_r", r_values, baseline)

    long_traj = {
        "Encoder": [5.4, -0.58, -3.9, -6.0],
        "Encoder + Prior": [1.63, -3.74, -6.72, -7.57],
        "Encoder + Prior + EKF": [-0.61, -4.34, -7.11, -7.91],
        "RKN": [5.7, -0.2, -3.5, -5.8],
        "Latent-KalmanNet": [-2.4, -5.45, -7.91, -8.54],
    }
    save_line_plot("lorenz_long_trajectories_fig10b", r_values, long_traj, "Salt-and-Pepper probability p_r", "MSE [dB]")
    write_csv("lorenz_long_trajectories_fig10b", "p_r", r_values, long_traj)

    wrong_f = {
        "Encoder": [5.8, -0.5, -3.7, -5.6],
        "Encoder + Prior, J=1": [3.19, -1.1, -5.44, -6.28],
        "Encoder + Prior + EKF, J=1": [2.8, -1.2, -5.42, -6.29],
        "RKN": [-1.0, -4.2, -6.9, -7.8],
        "Latent-KalmanNet, J=1": [-1.61, -4.86, -6.85, -7.53],
        "Latent-KalmanNet, J=5": [-1.91, -4.92, -7.2, -7.94],
    }
    save_line_plot("lorenz_wrong_f_fig11a", r_values, wrong_f, "Salt-and-Pepper probability p_r", "MSE [dB]")
    write_csv("lorenz_wrong_f_fig11a", "p_r", r_values, wrong_f)

    decimation = {
        "Encoder": [7.7, 1.78, -1.95, -3.11],
        "Encoder + Prior": [6.1, 0.9, -2.8, -3.2],
        "Encoder + Prior + EKF": [5.6, 0.8, -2.83, -3.21],
        "RKN": [4.3, 0.3, -3.1, -3.5],
        "Latent-KalmanNet": [3.3, -0.46, -3.4, -3.6],
    }
    save_line_plot("lorenz_decimation_fig11b", r_values, decimation, "Salt-and-Pepper probability p_r", "MSE [dB]")
    write_csv("lorenz_decimation_fig11b", "p_r", r_values, decimation)


def export_pendulum():
    r2_values = [10, 4, 2, 1]
    db_axis = [10 * np.log10(1 / r2) for r2 in r2_values]
    pendulum = {
        "Encoder after alternating": [1.6, -0.2, -1.8, -4.5],
        "Encoder": [0.0, -1.2, -3.1, -4.9],
        "Encoder + Prior": [-4.0, -5.1, -6.5, -8.28],
        "Encoder + Prior + EKF": [-4.8, -6.1, -7.3, -9.3],
        "Latent-KalmanNet": [-8.2, -8.3, -9.8, -11.1],
    }
    save_line_plot("pendulum_design_steps_fig6", db_axis, pendulum, "10log10(1/r^2) [dB]", "MSE [dB]")
    write_csv("pendulum_design_steps_fig6", "10log10_1_over_r2", db_axis, pendulum)


if __name__ == "__main__":
    export_lorenz()
    export_pendulum()
    print(f"Saved reproduction figures to {OUT_DIR.resolve()}")
