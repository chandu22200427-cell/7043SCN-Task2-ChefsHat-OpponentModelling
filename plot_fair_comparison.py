
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"

INPUT_PATH = RESULTS_DIR / "fair_comparison_summary.csv"

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Cannot find {INPUT_PATH}. Run fair_comparison.py first."
    )

df = pd.read_csv(INPUT_PATH)

required = {
    "agent_type",
    "configuration",
    "mean_seed_win_rate",
    "sd_seed_win_rate",
    "mean_seed_score",
    "mean_seed_position",
}
missing = required - set(df.columns)

if missing:
    raise ValueError(
        f"Missing columns: {sorted(missing)}\n"
        f"Found: {df.columns.tolist()}"
    )

config_order = ["all_random", "mixed", "all_rule_based"]
agent_order = ["RandomBaseline", "DQN"]

df["configuration"] = pd.Categorical(
    df["configuration"],
    categories=config_order,
    ordered=True,
)
df["agent_type"] = pd.Categorical(
    df["agent_type"],
    categories=agent_order,
    ordered=True,
)
df = df.sort_values(["configuration", "agent_type"])

# Print the summary used for the charts.
print("\nFair comparison summary:")
print(df.round(3).to_string(index=False))

metrics = [
    (
        "mean_seed_win_rate",
        "sd_seed_win_rate",
        "Mean win rate across seeds",
        "Win rate",
        "fair_comparison_win_rate.png",
        (0, 1),
    ),
    (
        "mean_seed_score",
        None,
        "Mean match score across seeds",
        "Average match score",
        "fair_comparison_average_score.png",
        None,
    ),
    (
        "mean_seed_position",
        None,
        "Mean finishing position across seeds",
        "Average finishing position (0 = first)",
        "fair_comparison_finishing_position.png",
        None,
    ),
]

x = np.arange(len(config_order))
width = 0.35

for value_col, error_col, title, ylabel, filename, ylim in metrics:
    plt.figure(figsize=(9, 5))

    for i, agent_name in enumerate(agent_order):
        subset = (
            df[df["agent_type"] == agent_name]
            .set_index("configuration")
            .reindex(config_order)
        )

        positions = x + (i - 0.5) * width
        errors = (
            subset[error_col].to_numpy()
            if error_col is not None
            else None
        )

        plt.bar(
            positions,
            subset[value_col].to_numpy(),
            width,
            yerr=errors,
            capsize=4 if errors is not None else 0,
            label=agent_name,
        )

    plt.xticks(x, config_order, rotation=10)
    plt.xlabel("Opponent configuration")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    if ylim is not None:
        plt.ylim(*ylim)

    plt.tight_layout()
    output_path = RESULTS_DIR / filename
    plt.savefig(output_path, dpi=300)
    plt.close()
    print("Saved:", output_path)

print("\nChart generation complete.")
