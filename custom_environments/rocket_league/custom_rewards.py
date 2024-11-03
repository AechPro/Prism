from rlgym_sim.utils.reward_functions import RewardFunction
from rlgym_sim.utils import common_values
import numpy as np


class InAirReward(RewardFunction): # We extend the class "RewardFunction"
    # Empty default constructor (required)
    def __init__(self):
        super().__init__()

    # Called when the game resets (i.e. after a goal is scored)
    def reset(self, initial_state):
        pass # Don't do anything when the game resets

    # Get the reward for a specific player, at the current state
    def get_reward(self, player, state, previous_action):

        # "player" is the current player we are getting the reward of
        # "state" is the current state of the game (ball, all players, etc.)
        # "previous_action" is the previous inputs of the player (throttle, steer, jump, boost, etc.) as an array

        if not player.on_ground:
            # We are in the air! Return full reward
            return 1
        else:
            # We are on ground, don't give any reward
            return 0


class TouchBallReward(RewardFunction):
    def __init__(self):
        super().__init__()
        self.prev_ball_vel = None

    def reset(self, initial_state):
        self.prev_ball_vel = None

    def get_reward(self, player, state, previous_action):
        if player.team_num == common_values.ORANGE_TEAM:
            ball = state.inverted_ball
        else:
            ball = state.ball

        if self.prev_ball_vel is None:
            self.prev_ball_vel = ball.linear_velocity
            return 0

        if player.ball_touched:
            diff_vel = ball.linear_velocity - self.prev_ball_vel

            # ||vel|| / max_speed can be at most 1.
            ball_accel = np.linalg.norm(diff_vel) / common_values.BALL_MAX_SPEED

            # Map accel to [-5, 5] for sigmoid.
            ball_accel = 10 * (ball_accel - 0.5)

            # Squish accel with sigmoid.
            reward = 1 / (1 + np.exp(-ball_accel))
        else:
            reward = 0

        self.prev_ball_vel = ball.linear_velocity
        return reward
