import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_convergence(csv_filename):
    # Load CSV
    df = pd.read_csv(csv_filename)

    # Extract data
    h = df["point_dist"]
    err_mean = df["err_mean"]
    err_max = df["err_max"]
    order_mean = df["err_mean_convergence_order"]
    order_max = df["err_max_convergence_order"]

    # Plot setup
    plt.figure(figsize=(12, 9))
    plt.loglog(h, err_mean, 'o-', label=r"$L_2$ error (mean)", markersize=10)
    plt.loglog(h, err_max, 's-', label=r"$L_\infty$ error (max)", markersize=10)

    # Annotate convergence rates
    for i in range(len(h) - 1):
        x_annot = np.sqrt(h.iloc[i] * h.iloc[i + 1])
        y_annot_mean = np.sqrt(err_mean.iloc[i] * err_mean.iloc[i + 1])
        y_annot_max = np.sqrt(err_max.iloc[i] * err_max.iloc[i + 1])

        plt.annotate(f"{order_mean.iloc[i]:.2f}", xy=(x_annot, y_annot_mean),
                     textcoords="offset points", xytext=(0, 20), ha='center', fontsize=28, color='blue')
        plt.annotate(f"{order_max.iloc[i]:.2f}", xy=(x_annot, y_annot_max),
                     textcoords="offset points", xytext=(0, -30), ha='center', fontsize=28, color='orange')

    # Labels and styling
    plt.xlabel("Discretization length $h$", fontsize=32)
    plt.ylabel("Relative error", fontsize=32)
    plt.title("Convergence of RBF Approximation", fontsize=36)
    plt.grid(True, which="both", ls="--")
    plt.legend(fontsize=28)

    # Explicit major x-ticks only
    tick_positions = h
    tick_labels = [f"{val:.3f}" for val in h]
    plt.xticks(ticks=tick_positions, labels=tick_labels, fontsize=28)

    # Remove minor x-ticks (both visually and logically)
    plt.tick_params(axis='x', which='minor', bottom=False, length=0)
    plt.gca().xaxis.set_minor_locator(plt.NullLocator())

    plt.yticks(fontsize=28)
    plt.tight_layout()
    plt.savefig(csv_filename.rstrip("csv") + "png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot convergence rates from a CSV file.")
    parser.add_argument("csv_filename", type=str, help="Path to the CSV file containing convergence data")
    args = parser.parse_args()

    plot_convergence(args.csv_filename)
