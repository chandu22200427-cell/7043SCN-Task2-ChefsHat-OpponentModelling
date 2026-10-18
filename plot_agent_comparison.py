
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

input_path = RESULTS / "opponent_aware_evaluation_summary.csv"
output_path = RESULTS / "agent_comparison_win_rate.png"

df = pd.read_csv(input_path)

required = {
    "agent_type",
    "configuration",
    "mean_seed_win_rate",
}
if not required.issubset(df.columns):
    raise ValueError(f"CSV must contain these columns: {required}")

pivot = df.pivot(
    index="configuration",
    columns="agent_type",
    values="mean_seed_win_rate",
)

configuration_order = [
    "all_random",
    "mixed",
    "all_rule_based",
]
agent_order = [
    "RandomBaseline",
    "DQN",
    "OpponentAwareDQN",
]

pivot = pivot.reindex(index=configuration_order)
pivot = pivot.reindex(
    columns=[name for name in agent_order if name in pivot.columns]
)

ax = (pivot * 100).plot(
    kind="bar",
    figsize=(10, 6),
    width=0.8,
)

ax.set_title("Agent Win Rate Across Opponent Configurations")
ax.set_xlabel("Opponent configuration")
ax.set_ylabel("Mean win rate across seeds (%)")
ax.set_ylim(0, 100)
ax.set_xticklabels(
    ["All random", "Mixed", "All rule-based"],
    rotation=0,
)
ax.legend(title="Agent")
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(output_path, dpi=200, bbox_inches="tight")
plt.close()

print(f"Chart saved to: {output_path}")
