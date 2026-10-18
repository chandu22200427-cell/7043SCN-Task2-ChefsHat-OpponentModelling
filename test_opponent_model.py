
from opponent_model import OpponentModel


def main():
    model = OpponentModel(["Agent2", "Agent3", "Agent4"])

    model.update("Agent2", 199)  # Pass
    model.update("Agent2", 10)   # Play
    model.update("Agent2", 11)   # Play

    profile = model.get_profile("Agent2")

    print("Agent2 profile:", profile)
    print("State features:", model.get_state_features("Agent2"))

    assert profile["observed_actions"] == 3
    assert profile["pass_rate"] == 2 / 5
    assert profile["play_rate"] == 3 / 5

    print("Opponent model test passed.")


if __name__ == "__main__":
    main()
