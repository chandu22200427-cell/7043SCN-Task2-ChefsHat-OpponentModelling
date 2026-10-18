
import numpy as np
import torch

from dqn_agent import DQNAgent
from environment_check import create_fixed_match_environment
from opponent_model import OpponentModel
from train_opponent_aware import (
    augment_observation,
    OPPONENT_NAMES,
    STATE_SIZE,
)


def main():
    env, observation, _ = create_fixed_match_environment(seed=42)

    try:
        model = OpponentModel(OPPONENT_NAMES)
        state = augment_observation(observation, model)

        agent = DQNAgent(
            state_size=STATE_SIZE,
            action_size=200,
            epsilon=0.0,
            seed=42,
        )

        action = agent.select_action(state)
        legal_actions = np.flatnonzero(observation[28:228] == 1)

        print("PyTorch version:", torch.__version__)
        print("State shape:", state.shape)
        print("Selected action:", action)
        print("Legal actions:", legal_actions.tolist())

        assert state.shape == (234,)
        assert action in legal_actions

        print("Opponent-aware DQN test passed.")

    finally:
        env.close()


if __name__ == "__main__":
    main()
