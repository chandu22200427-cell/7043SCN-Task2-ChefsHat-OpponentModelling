
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"

BASELINE_PATH = RESULTS_DIR / "opponent_comparison.csv"
DQN_PATH = RESULTS_DIR / "dqn_evaluation.csv"

if not BASELINE_PATH.exists():
    raise FileNotFoundError(f"Baseline file not found: {BASELINE_PATH}")

if not DQN_PATH.exists():
    raise FileNotFoundError(f"DQN evaluation file not found: {DQN_PATH}")

baseline = pd.read_csv(BASELINE_PATH)
dqn = pd.read_csv(DQN_PATH)

# Summarise the random-action baseline.
baseline_summary = (
    baseline.groupby("configuration")
    .agg(
        baseline_matches=("agent1_position", "count"),
        baseline_wins=("agent1_position", lambda x: (x == 0).sum()),
        baseline_win_rate=("agent1_position", lambda x: (x == 0).mean()),
        baseline_average_score=("agent1_match_score", "mean"),
        baseline_average_position=("agent1_position", "mean"),
    )
    .reset_index()
)

# Summarise the trained DQN evaluation.
dqn_summary = (
    dqn.groupby("configuration")
    .agg(
        dqn_matches=("agent1_position", "count"),
        dqn_wins=("agent1_position", lambda x: (x == 0).sum()),
        dqn_win_rate=("agent1_position", lambda x: (x == 0).mean()),
        dqn_average_score=("agent1_match_score", "mean"),
        dqn_average_position=("agent1_position", "mean"),
    )
    .reset_index()
)

summary = baseline_summary.merge(dqn_summary, on="configuration")

summary["win_rate_difference_percentage_points"] = (
    summary["dqn_win_rate"] - summary["baseline_win_rate"]
) * 100

summary_path = RESULTS_DIR / "baseline_vs_dqn_summary.csv"
summary.to_csv(summary_path, index=False)

print("\nBaseline vs DQN comparison:")
print(summary.round(3).to_string(index=False))

# Chart 1: Win-rate comparison.
x = np.arange(len(summary))
width = 0.35

plt.figure(figsize=(9, 5))
plt.bar(
    x - width / 2,
    summary["baseline_win_rate"],
    width,
    label="Random-action baseline",
)
plt.bar(
    x + width / 2,
    summary["dqn_win_rate"],
    width,
    label="Trained DQN (greedy)",
)
plt.xticks(x, summary["configuration"], rotation=10)
plt.ylabel("Win rate")
plt.xlabel("Opponent configuration")
plt.title("Random Baseline vs Trained DQN: Win Rate")
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "baseline_vs_dqn_win_rate.png", dpi=300)
plt.close()

# Chart 2: Average-score comparison.
plt.figure(figsize=(9, 5))
plt.bar(
    x - width / 2,
    summary["baseline_average_score"],
    width,
    label="Random-action baseline",
)
plt.bar(
    x + width / 2,
    summary["dqn_average_score"],
    width,
    label="Trained DQN (greedy)",
)
plt.xticks(x, summary["configuration"], rotation=10)
plt.ylabel("Average match score")
plt.xlabel("Opponent configuration")
plt.title("Random Baseline vs Trained DQN: Average Score")
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "baseline_vs_dqn_average_score.png", dpi=300)
plt.close()

# Chart 3: Average finishing position (lower is better).
plt.figure(figsize=(9, 5))
plt.bar(
    x - width / 2,
    summary["baseline_average_position"],
    width,
    label="Random-action baseline",
)
plt.bar(
    x + width / 2,
    summary["dqn_average_position"],
    width,
    label="Trained DQN (greedy)",
)
plt.xticks(x, summary["configuration"], rotation=10)
plt.ylabel("Average finishing position (0 = first)")
plt.xlabel("Opponent configuration")
plt.title("Random Baseline vs Trained DQN: Finishing Position")
plt.legend()
plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "baseline_vs_dqn_finishing_position.png",
    dpi=300,
)
plt.close()

print("\nSaved comparison outputs:")
print(summary_path)
print(RESULTS_DIR / "baseline_vs_dqn_win_rate.png")
print(RESULTS_DIR / "baseline_vs_dqn_average_score.png")
print(RESULTS_DIR / "baseline_vs_dqn_finishing_position.png")
