from typing import Optional
import numpy as np


class RocketSimEnv(object):
    def __init__(self, render_this_env=False):
        self.env = self._build_env()
        self.n_actions = self.env.action_space.n
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.render_this_env = render_this_env

    def step(self, actions):
        if self.render_this_env:
            self.render()

        obs, rew, done, trunc, info = self.env.step(actions)
        return np.asarray(obs), rew, done, trunc, info

    def reset(self, return_info=True, seed=123):
        self._set_seed(seed)
        obs, info = self.env.reset(return_info=True)
        return np.asarray(obs), info

    def _set_seed(self, seed: Optional[int]):
        pass

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def _build_env(self):
        import rlgym_sim
        from rlgym_sim.utils.reward_functions import CombinedReward
        from rlgym_sim.utils.reward_functions.common_rewards import VelocityPlayerToBallReward, \
            VelocityBallToGoalReward, EventReward, FaceBallReward

        from rlgym_sim.utils.obs_builders import DefaultObs
        from rlgym_sim.utils.terminal_conditions.common_conditions import NoTouchTimeoutCondition, \
            GoalScoredCondition
        from custom_environments.rocket_league.custom_rewards import InAirReward, TouchBallReward
        from rlgym_sim.utils import common_values
        from rlgym_tools.extra_action_parsers.lookup_act import LookupAction
        spawn_opponents = True
        team_size = 1
        game_tick_rate = 120
        tick_skip = 8
        timeout_seconds = 30
        timeout_ticks = int(round(timeout_seconds * game_tick_rate / tick_skip))

        action_parser = LookupAction()
        terminal_conditions = [NoTouchTimeoutCondition(timeout_ticks), GoalScoredCondition()]

        reward_scale = 0.1
        touch_scale = 0.05
        rewards = (
            (EventReward(touch=0.0, goal=1, concede=-1), 50 * reward_scale),
            (TouchBallReward(), 50 * touch_scale * reward_scale),
            (VelocityPlayerToBallReward(positive_only=True), 2.5 * reward_scale),
            (FaceBallReward(), 0.25 * reward_scale),
            (InAirReward(), 0.15 * reward_scale),
            (VelocityBallToGoalReward(), 5 * reward_scale)
        )

        reward_fn = CombinedReward.from_zipped(*rewards)

        obs_builder = DefaultObs(
            pos_coef=np.asarray(
                [1 / common_values.SIDE_WALL_X, 1 / common_values.BACK_NET_Y, 1 / common_values.CEILING_Z]),
            ang_coef=1 / np.pi,
            lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
            ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL)

        env = rlgym_sim.make(tick_skip=tick_skip,
                             team_size=team_size,
                             spawn_opponents=spawn_opponents,
                             terminal_conditions=terminal_conditions,
                             reward_fn=reward_fn,
                             obs_builder=obs_builder,
                             action_parser=action_parser)

        return env
