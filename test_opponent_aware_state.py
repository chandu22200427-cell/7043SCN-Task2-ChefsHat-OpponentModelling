
import numpy as np

from environment_check import create_fixed_match_environment
from opponent_model import OpponentModel

BASE_STATE_SIZE = 228
STATE_SIZE = 234
OPPONENT_NAMES = ("Agent2", "Agent3", "Agent4")


def augment_observation(observation, opponent_model):
    base = np.asarray(observation, dtype=np.float32)

    if base.shape != (BASE_STATE_SIZE,):
        raise ValueError(f"Unexpected observation shape: {base.shape}")

    features = []
    for name in OPPONENT_NAMES:
        features.extend(opponent_model.get_state_features(name))

    augmented = np.concatenate([
        base,
        np.asarray(features, dtype=np.float32),
    ])

    if augmented.shape != (STATE_SIZE,):
        raise ValueError(f"Unexpected augmented shape: {augmented.shape}")

    return augmented


def main():
    env, observation, _ = create_fixed_match_environment(seed=42)

    try:
        model = OpponentModel(OPPONENT_NAMES)
        augmented = augment_observation(observation, model)

        print("Original observation shape:", observation.shape)
        print("Augmented observation shape:", augmented.shape)
        print("Expected state size:", STATE_SIZE)

        assert observation.shape == (BASE_STATE_SIZE,)
        assert augmented.shape == (234,)

        # The original legal-action mask must remain at indices 28:228.
        legal_actions = np.flatnonzero(augmented[28:228] == 1)
        print("Number of legal actions:", len(legal_actions))

        assert len(legal_actions) > 0
        assert np.all(augmented[228:] == 0.5)

        print("Augmented-state test passed.")

    finally:
        env.close()


if __name__ == "__main__":
    main()
