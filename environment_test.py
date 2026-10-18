
from pathlib import Path

import gym
import numpy as np
import ChefsHatGym.env


LOG_DIRECTORY = str(
    (Path(__file__).resolve().parent / "results" / "chefshat_experiment")
)


def create_environment(seed=42):
    """Create and initialize the official Chef's Hat environment."""
    env = gym.make("chefshat-v1")

    env.startExperiment(
        playerNames=["Agent1", "Agent2", "Agent3", "Agent4"],
        logDirectory=LOG_DIRECTORY,
    )

    observation, info = env.reset(seed=seed)

    return env, observation, info


def get_legal_actions(observation):
    """Return indices of currently legal actions."""
    action_mask = observation[28:228]
    return np.flatnonzero(action_mask == 1)


def submit_action(env, action_index):
    """Submit one selected action in the required 200-value format."""
    action = np.zeros(200, dtype=np.float32)
    action[action_index] = 1.0

    return env.step(action)


if __name__ == "__main__":
    env, observation, _ = create_environment()

    try:
        legal_actions = get_legal_actions(observation)

        print("Environment initialized successfully")
        print("Observation shape:", observation.shape)
        print("Number of legal actions:", len(legal_actions))
        print("Legal action indices:", legal_actions)

        if len(legal_actions) == 0:
            raise RuntimeError("No legal actions were found.")

        next_observation, reward, terminated, truncated, info = (
            submit_action(env, int(legal_actions[0]))
        )

        print("Action valid:", info.get("Action_Valid"))
        print("Action description:", info.get("Action_Decoded"))
        print("Next observation shape:", next_observation.shape)
        print("Immediate reward:", reward)

    finally:
        env.close()
