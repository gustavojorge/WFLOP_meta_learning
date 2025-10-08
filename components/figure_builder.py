import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Tuple

def normalize_importance(values):
    return (values - np.min(values)) / (np.max(values) - np.min(values))

def plot_feature_importance(ax, df, title):
    df['normalized_importance'] = normalize_importance(df['Importance'])

    df = df.sort_values(by='normalized_importance', ascending=False).head(10)

    palette = sns.color_palette("tab10", len(df))
    
    for i, (feature, importance) in enumerate(zip(df['Feature'], df['normalized_importance'])):
        ax.plot([0, importance], [feature, feature], linestyle='-', color=palette[i], linewidth=2)
    
    # Usamos scatter direto no eixo para poder passar uma lista de cores sem precisar de 'hue'
    ax.scatter(df['normalized_importance'], df['Feature'], marker='o', s=70, c=palette, zorder=3)
    
    ax.set_xlim(0, 1)
    ax.set_xlabel("Normalized Importance", fontsize=18, labelpad=6)
    ax.set_ylabel("Feature", fontsize=18, labelpad=10)
    ax.set_title(title, fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.grid(False)

def load_feature_importance(label, l_value, r_value, adaptive_walk=False):
    if adaptive_walk:
        file_path = f"result/{label}/features_importance/r{r_value}_features_importance.csv"
    else:
        file_path = f"result/{label}/features_importance/l{l_value}_r{r_value}_features_importance.csv"
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Feature importance file not found: {file_path}")
    df = pd.read_csv(file_path)
    if 'Feature' not in df.columns or 'Importance' not in df.columns:
        raise ValueError(f"Invalid feature importance schema in {file_path}")
    return df

def find_optimal_parameters(label) -> Tuple[str, str, str]:
    merit_path = f"result/{label}/merit_{label}.csv"
    if not os.path.isfile(merit_path):
        raise FileNotFoundError(f"Merit table not found: {merit_path}. Run merit_table_builder first.")
    dataset = pd.read_csv(merit_path, index_col=0)
    if dataset.empty:
        raise ValueError("Merit table is empty.")
    # Tenta converter valores para float ignorando erros (Erros vira NaN)
    numeric = dataset.apply(pd.to_numeric, errors='coerce')
    if numeric.isna().all().all():
        raise ValueError("All merit values are non-numeric (only errors?).")
    min_merit_idx = np.unravel_index(np.nanargmin(numeric.values, axis=None), numeric.shape)
    r_opt, l_opt = numeric.index[min_merit_idx[0]], numeric.columns[min_merit_idx[1]]
    r_opt_label = r_opt.replace("r=", "").replace("%", "").strip()
    r_mapping = {"100": "1.0", "50": "0.5", "25": "0.25", "10": "0.1", "5": "0.05"}
    r_opt = r_mapping.get(r_opt_label, r_opt_label)
    l_opt = l_opt.replace("ℓ = ", "").strip()
    return l_opt, r_opt, r_opt_label

def save_figure(fig, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")

def main(label):
    output_dir = f"result/{label}/figures/"
    os.makedirs(output_dir, exist_ok=True)
    try:
        df_left = load_feature_importance(label, l_value=100, r_value="1.0")
    except Exception as e:
        raise RuntimeError(f"Failed to load baseline feature importance (l100_r1.0): {e}")

    l_opt, r_opt, r_opt_label = find_optimal_parameters(label)
    adaptive_walk = (l_opt == "Adaptive Walk")
    try:
        df_right = load_feature_importance(label, l_opt, r_opt, adaptive_walk)
    except Exception as e:
        raise RuntimeError(f"Failed to load optimal configuration feature importance (l={l_opt}, r={r_opt}): {e}")

    print("-> Building the main figure")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    plot_feature_importance(axes[0], df_left, "(length=100, neighborhood=100%)")
    plot_feature_importance(axes[1], df_right, f"(length={l_opt}, neighborhood={r_opt_label})" if not adaptive_walk else f"({l_opt}, neighborhood={r_opt_label})")
    plt.tight_layout()
    save_figure(fig, output_dir, f"feature_importance_{label}.png")

    for side, df, filename in [
        ("left", df_left, "left_figure_length_100_neighborhood_100.png"),
        ("right", df_right, (
            f"right_figure_length_{l_opt}_neighborhood_{r_opt}.png" if not adaptive_walk else f"right_figure_AdaptiveWalk_neighborhood_{r_opt}.png"
        )),
    ]:
        print(f"--> Building the {side} figure")
        fig_side, ax_side = plt.subplots(figsize=(6, 6))
        plot_feature_importance(ax_side, df, "")
        plt.tight_layout()
        save_figure(fig_side, output_dir, filename)
    
if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        raise ValueError("!!! ERROR (figure_builder.py) -> No label provided. !!!")
