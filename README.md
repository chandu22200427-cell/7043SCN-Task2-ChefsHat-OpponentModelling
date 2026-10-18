# Chef's Hat Gym --- DQN with Opponent-Configuration Experiments

**Module:** 7043SCN --- Generative AI and Reinforcement Learning\
**Task:** Task 2 --- Reinforcement Learning\
**Variant:** Opponent Modelling\
**Environment:** [Chef's Hat
Gym](https://github.com/pablovin/ChefsHatGYM)

## 1. Project overview

This project investigates whether a Deep Q-Network (DQN) agent can
perform better than a random legal-action baseline in the competitive,
turn-based Chef's Hat card game. The Opponent Modelling variant is
explored by evaluating agents against three opponent configurations:

-   **All random:** all three opponents choose random legal actions.
-   **Mixed:** one opponent is random and two use a simple rule-based
    policy.
-   **All rule-based:** all three opponents use the simple rule-based
    policy.

The experiments compare win rate, match score and finishing position.
**Scope note:** the current DQN does not contain a separate learned
opponent classifier or opponent-intention model. It studies performance
across opponent configurations rather than implementing a fully explicit
opponent-modelling architecture.

## 2. Research question

**How does a trained DQN compare with a random legal-action baseline
against different opponent-policy configurations in Chef's Hat Gym?**

## 3. Environment and interface

The project uses the Gym-compatible `chefshat-v1` environment.

### Observation

The tested environment returns a 228-element observation vector:

  Indices      Length Meaning
  ---------- -------- -------------------------------------
  `0:11`           11 Board-related features
  `11:28`          17 Current-hand features
  `28:228`        200 Legal-action mask (`1` means legal)

### Action handling

Although the environment advertises `Discrete(200)`, its tested `step()`
method expects a 200-element action vector. The chosen action is encoded
as a one-hot vector. The DQN outputs 200 Q-values and masks illegal
actions during selection and next-state value calculation.

### Reward

The environment's immediate reward was observed to be zero during
ordinary steps. Training therefore uses a match-outcome reward based on
the agent's zero-based finishing position:

    Position Outcome     Match score   Training reward
  ---------- --------- ------------- -----------------
           0 First                 3             1.000
           1 Second                2             0.667
           2 Third                 1             0.333
           3 Fourth                0             0.000

The reward is match score divided by 3. Because reward is delayed until
the match ends, credit assignment is challenging. The training
implementation treats the interval between two decisions by the learning
agent as one transition and discounts by the number of intervening
environment steps; this is an approximation in a multi-agent turn-based
game.

## 4. Agent and algorithm

The learning agent is a **Deep Q-Network (DQN)** implemented with
PyTorch.

-   Input: 228 observation features.
-   Hidden layers: 256 and 128 units with ReLU activations.
-   Output: 200 Q-values.
-   Legal-action masking to avoid invalid choices.
-   Experience replay buffer: capacity 20,000 transitions.
-   Epsilon-greedy exploration.
-   Smooth L1 (Huber) loss and gradient clipping at 1.0.
-   Target network, periodically updated during training.
-   Device: CPU in the current setup.

DQN is a reasonable baseline because the observation is numerical and
the environment exposes a fixed set of 200 action indices. The
legal-action mask is essential because only a subset of actions is valid
in any state.

### Training configuration

  Setting                                 Value
  -------------------------------- ------------
  Random seed                                42
  Training matches                          100
  Maximum environment steps             100,000
  Target-network update interval     10 matches
  Replay capacity                        20,000
  Device                                    CPU

See `dqn_agent.py` for the exact exploration and discount settings. The
training script rotates among the three opponent configurations and
writes logs and a learning curve to `results/`.

## 5. Opponent configurations

  Configuration      Opponent 2   Opponent 3   Opponent 4
  ------------------ ------------ ------------ ------------
  `all_random`       Random       Random       Random
  `mixed`            Random       Rule-based   Rule-based
  `all_rule_based`   Rule-based   Rule-based   Rule-based

The random policy samples from legal actions. The rule-based policy
chooses the first available legal action other than pass, if possible;
otherwise it passes. This is a simple baseline, not an expert strategy.

## 6. Setup and reproducibility

### Requirements

-   Python 3.11 (tested with Python 3.11.9)
-   Git
-   Packages listed in `requirements.txt`, if present

### Windows PowerShell

``` powershell
git clone https://github.com/chandu22200427-cell/7043SCN-Task2-ChefsHat-OpponentModelling.git
cd 7043SCN-Task2-ChefsHat-OpponentModelling
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install chefshatgym
python -m pip install "gym==0.26.2"
python -m pip install numpy pandas matplotlib torch
```

The installed package is imported in Python as `ChefsHatGym`
(capitalisation matters).

### Run experiments

Run commands from the repository root with the virtual environment
activated. Confirm each script is present in the final repository before
running it.

``` powershell
python environment_test.py
python evaluate_opponents.py
python plot_opponent_results.py
python train_dqn.py
python fair_comparison.py
python plot_fair_comparison.py
```

The training script saves `models/dqn_opponent_modeling.pt`. CSV files
and plots are saved under `results/`. Output files depend on which
scripts have been run.

## 7. Evaluation methodology

The trained DQN is evaluated greedily (epsilon = 0) and compared with
the random legal-action baseline. Three seeds are used: **42, 123 and
2026**. Each agent/configuration/seed combination contains **30
matches**, for 540 matches total (2 agent types × 3 configurations × 3
seeds × 30 matches).

The same seed values are used for both agent types, but different
actions cause game trajectories to diverge; this is not a strictly
paired-trajectory experiment. The reported standard deviations are
across the three seed-level win rates, not confidence intervals.

### Latest recorded results

  ----------------------------------------------------------------------------------
  Agent      Opponent           Matches   Mean win SD of seed Mean score        Mean
             configuration                    rate  win rates              finishing
                                            across                          position
                                             seeds                       
  ---------- --------------- ---------- ---------- ---------- ---------- -----------
  Random     All random              90      0.256      0.051      1.589       1.411
  baseline                                                               

  DQN        All random              90      0.611      0.019      2.378       0.622

  Random     Mixed                   90      0.311      0.051      1.678       1.322
  baseline                                                               

  DQN        Mixed                   90      0.556      0.184      2.244       0.756

  Random     All rule-based          90      0.633      0.088      2.289       0.711
  baseline                                                               

  DQN        All rule-based          90      0.744      0.126      2.522       0.478
  ----------------------------------------------------------------------------------

Lower average finishing position is better because position 0 is first
place. Win rate is the proportion of matches in which the agent finishes
first.

**Interpretation:** In this recorded run, the DQN has a higher mean win
rate and mean score than the random baseline in all three
configurations. However, DQN performance varies across seeds, especially
in the mixed and all-rule-based settings. These results do not prove
that it will consistently outperform every opponent policy. Re-run the
evaluation before submission if the code, checkpoint or environment
settings change.

## 8. Experimental outputs

The scripts are intended to produce these files, depending on which
experiments have been run:

  --------------------------------------------------------------------------------------
  File                                               Purpose
  -------------------------------------------------- -----------------------------------
  `results/opponent_comparison.csv`                  Per-match baseline opponent results

  `results/opponent_comparison_summary.csv`          Baseline comparison summary

  `results/opponent_win_rate.png`                    Baseline win-rate plot

  `results/opponent_average_score.png`               Baseline average-score plot

  `results/opponent_finishing_position.png`          Baseline finishing-position plot

  `results/dqn_training_log.csv`                     Training match-level log

  `results/dqn_training_summary.csv`                 Training summary by opponent
                                                     configuration

  `results/dqn_training_curve.png`                   Training curve

  `models/dqn_opponent_modeling.pt`                  Saved DQN checkpoint

  `results/fair_comparison_raw.csv`                  Match-level evaluation results

  `results/fair_comparison_summary.csv`              Aggregated evaluation summary

  `results/fair_comparison_win_rate.png`             DQN vs baseline win rate

  `results/fair_comparison_average_score.png`        DQN vs baseline average score

  `results/fair_comparison_finishing_position.png`   DQN vs baseline finishing position
  --------------------------------------------------------------------------------------

Commit only outputs that exist and are intended for submission. Do not
commit `.venv/` or cache files.

## 9. Project structure

Expected layout (update this list to match the actual final repository):

``` text
7043SCN-Task2-ChefsHat-OpponentModelling/
├── agents.py
├── dqn_agent.py
├── environment_check.py
├── environment_test.py
├── evaluate_opponents.py
├── train_dqn.py
├── evaluate_dqn.py
├── fair_comparison.py
├── plot_opponent_results.py
├── plot_fair_comparison.py
├── models/
│   └── dqn_opponent_modeling.pt
├── results/
│   ├── *.csv
│   └── *.png
├── requirements.txt
└── README.md
```

## 10. Limitations and critical reflection

1.  **Limited training budget:** 100 training matches is a small budget.
    More training, multiple independent training runs and systematic
    hyperparameter experiments would improve the strength of the
    conclusions.
2.  **Sparse terminal reward:** delayed reward makes it difficult to
    assign credit to earlier decisions.
3.  **Limited opponent modelling:** the experiments vary opponent-policy
    configurations, but the DQN does not explicitly infer opponent types
    or learn an opponent-behaviour model.
4.  **Simple rule-based opponent:** the heuristic is not an expert
    strategy and may not represent human play or stronger learned
    agents.
5.  **Transition approximation:** the learning agent's next observation
    is obtained at its next decision after other players have acted.
    Discounting by intervening steps is an approximation.
6.  **Evaluation uncertainty:** only three seeds are used, and
    variability is visible in some configurations. The reported standard
    deviations are descriptive, not confidence intervals.
7.  **Environment compatibility:** the tested setup uses Gym 0.26.2 and
    emits a warning that the older Gym package is no longer maintained.
    Reproduce with documented versions and record compatibility changes.

## 11. Conclusion

In the current evaluation, DQN performs better on average than the
random legal-action baseline across the three tested opponent
configurations. The small training budget, sparse rewards, simplified
rule-based policy and seed variability limit the strength and generality
of this conclusion. The results motivate further experiments rather than
demonstrating universally robust performance.

## 12. AI assistance declaration

AI tools were used to help draft and review parts of the documentation
and to support debugging and explanation of implementation concepts. The
student should verify the implementation, environment behaviour, results
and conclusions, annotate AI-assisted code or text as required by the
module, and ensure they can explain the work independently. This
declaration should be adjusted to accurately describe the assistance
actually used and comply with the module's academic-integrity
requirements.

## References

-   Chef's Hat Gym source repository:
    https://github.com/pablovin/ChefsHatGYM
-   Chef's Hat Gym documentation:
    https://chefshatgym.readthedocs.io/en/latest/
-   PyTorch documentation: https://pytorch.org/docs/stable/
-   Gym documentation: https://www.gymlibrary.dev/
