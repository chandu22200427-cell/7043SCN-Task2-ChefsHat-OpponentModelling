
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"
CSV_PATH = RESULTS_DIR / "opponent_comparison.csv"

# Load the experiment results
if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"Results file not found: {CSV_PATH}\n"
        "Run evaluate_opponents.py first."
    )

df = pd.read_csv(CSV_PATH)

# Validate the actual columns in your CSV
required_columns = {
    "configuration",
    "match",
    "agent1_position",
    "agent1_match_score",
}

missing = required_columns - set(df.columns)
if missing:
    raise ValueError(
        f"Missing columns: {sorted(missing)}\n"
        f"Available columns: {df.columns.tolist()}"
    )

print(f"Loaded {len(df)} match records.")
print("\nRecords per configuration:")
print(df.groupby("configuration").size())

# Summarise results for each opponent configuration
summary = (
    df.groupby("configuration")
    .agg(
        matches=("match", "count"),
        wins=("agent1_position", lambda x: (x == 0).sum()),
        average_score=("agent1_match_score", "mean"),
        average_finishing_position=("agent1_position", "mean"),
    )
    .reset_index()
)

summary["win_rate"] = summary["wins"] / summary["matches"]

# Save summary table
summary_path = RESULTS_DIR / "opponent_comparison_summary.csv"
summary.to_csv(summary_path, index=False)

print("\nOpponent comparison summary:")
print(summary.to_string(index=False))

# Plot 1: Win rate
plt.figure(figsize=(8, 5))
plt.bar(summary["configuration"], summary["win_rate"])
plt.title("Agent1 Win Rate by Opponent Configuration")
plt.xlabel("Opponent configuration")
plt.ylabel("Win rate")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "opponent_win_rate.png", dpi=300)
plt.close()

# Plot 2: Average match score
plt.figure(figsize=(8, 5))
plt.bar(summary["configuration"], summary["average_score"])
plt.title("Agent1 Average Match Score by Opponent Configuration")
plt.xlabel("Opponent configuration")
plt.ylabel("Average match score")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "opponent_average_score.png", dpi=300)
plt.close()

# Plot 3: Average finishing position (0 = first place)
plt.figure(figsize=(8, 5))
plt.bar(
    summary["configuration"],
    summary["average_finishing_position"],
)
plt.title("Agent1 Average Finishing Position by Opponent Configuration")
plt.xlabel("Opponent configuration")
plt.ylabel("Average finishing position (lower is better)")
plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "opponent_finishing_position.png",
    dpi=300,
)
plt.close()

print("\nAll outputs saved in:", RESULTS_DIR)
print("1. opponent_comparison_summary.csv")
print("2. opponent_win_rate.png")
print("3. opponent_average_score.png")
print("4. opponent_finishing_position.png")
