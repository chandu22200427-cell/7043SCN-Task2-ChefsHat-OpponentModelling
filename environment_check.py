
from pathlib import Path

import gym
import ChefsHatGym.env


def create_fixed_match_environment(number_of_matches=5, seed=42):
    """Create Chef's Hat configured to stop after a fixed match count."""
    env = gym.make("chefshat-v1")

    log_directory = str(
        (Path(__file__).resolve().parent / "results" / "chefshat_experiment")
    )

    env.startExperiment(
        gameType="MATCHES",
        stopCriteria=number_of_matches,
        playerNames=["Agent1", "Agent2", "Agent3", "Agent4"],
        logDirectory=log_directory,
    )

    observation, info = env.reset(seed=seed)

    return env, observation, info


if __name__ == "__main__":
    env, observation, info = create_fixed_match_environment()

    try:
        print("Environment initialized.")
        print("Observation shape:", observation.shape)
        print("Initial match count:", env.unwrapped.matches)
        print("Stop criterion:", env.unwrapped.stopCriteria)
        print("Game type:", env.unwrapped.gameType)
    finally:
        env.close()
