"""Implementation of Atari 2600 Preprocessing following the guidelines of Machado et al., 2018."""
import numpy as np

import gymnasium as gym
from gymnasium.spaces import Box


try:
    import cv2
except ImportError:
    cv2 = None


class AtariPreprocessing(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """Atari 2600 preprocessing wrapper.

    This class follows the guidelines in Machado et al. (2018),
    "Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems for General Agents".

    Specifically, the following preprocess stages applies to the atari environment:

    - Noop Reset: Obtains the initial state by taking a random number of no-ops on reset, default max 30 no-ops.
    - Frame skipping: The number of frames skipped between steps, 4 by default
    - Max-pooling: Pools over the most recent two observations from the frame skips
    - Termination signal when a life is lost: When the agent losses a life during the environment, then the environment is terminated.
        Turned off by default. Not recommended by Machado et al. (2018).
    - Resize to a square image: Resizes the atari environment original observation shape from 210x180 to 84x84 by default
    - Grayscale observation: If the observation is colour or greyscale, by default, greyscale.
    - Scale observation: If to scale the observation between [0, 1) or [0, 255), by default, not scaled.
    """

    def __init__(
        self,
        env: gym.Env,
        noop_max: int = 30,
        frame_skip: int = 4,
        frame_stack: int = 4,
        screen_size: int = 84,
        terminal_on_life_loss: bool = False,
        grayscale_obs: bool = True,
        scale_obs: bool = False,
    ):
        """Wrapper for Atari 2600 preprocessing.

        Args:
            env (Env): The environment to apply the preprocessing
            noop_max (int): For No-op reset, the max number no-ops actions are taken at reset, to turn off, set to 0.
            frame_skip (int): The number of frames between new observation the agents observations effecting the frequency at which the agent experiences the game.
            screen_size (int): resize Atari frame
            terminal_on_life_loss (bool): `if True`, then :meth:`step()` returns `terminated=True` whenever a
                life is lost.
            grayscale_obs (bool): if True, then gray scale observation is returned, otherwise, RGB observation
                is returned.
            grayscale_newaxis (bool): `if True and grayscale_obs=True`, then a channel axis is added to
                grayscale observations to make them 3-dimensional.
            scale_obs (bool): if True, then observation normalized in range [0,1) is returned. It also limits memory
                optimization benefits of FrameStack Wrapper.

        Raises:
            DependencyNotInstalled: opencv-python package not installed
            ValueError: Disable frame-skipping in the original env
        """
        gym.utils.RecordConstructorArgs.__init__(
            self,
            noop_max=noop_max,
            frame_skip=frame_skip,
            screen_size=screen_size,
            terminal_on_life_loss=terminal_on_life_loss,
            grayscale_obs=grayscale_obs,
            grayscale_newaxis=True,
            scale_obs=scale_obs,
        )
        gym.Wrapper.__init__(self, env)

        if cv2 is None:
            raise gym.error.DependencyNotInstalled(
                "opencv-python package not installed, run `pip install gymnasium[other]` to get dependencies for atari"
            )
        assert frame_skip > 0
        assert screen_size > 0
        assert noop_max >= 0
        if frame_skip > 1:
            if (
                env.spec is not None
                and "NoFrameskip" not in env.spec.id
                and getattr(env.unwrapped, "_frameskip", None) != 1
            ):
                raise ValueError(
                    "Disable frame-skipping in the original env. Otherwise, more than one "
                    "frame-skip will happen as through this wrapper"
                )
        self.noop_max = noop_max
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

        self.frame_skip = frame_skip
        self.frame_stack = frame_stack
        self.screen_size = screen_size
        self.terminal_on_life_loss = terminal_on_life_loss
        self.grayscale_obs = grayscale_obs
        self.grayscale_newaxis = True
        self.scale_obs = scale_obs

        # buffer of most recent two observations for max pooling
        assert isinstance(env.observation_space, Box)
        if grayscale_obs:
            self.obs_buffer = [
                np.empty(env.observation_space.shape[:2], dtype=np.uint8),
                np.empty(env.observation_space.shape[:2], dtype=np.uint8),
            ]
            self.frame_pool_buffer = np.zeros(env.observation_space.shape[:2], dtype=np.uint8)

        else:
            self.obs_buffer = [
                np.empty(env.observation_space.shape, dtype=np.uint8),
                np.empty(env.observation_space.shape, dtype=np.uint8),
            ]
            self.frame_pool_buffer = np.zeros(env.observation_space.shape, dtype=np.uint8)

        self.lives = 0
        self.game_over = False

        _low, _high, _obs_dtype = (
            (0, 255, np.uint8) if not scale_obs else (0, 1, np.float32)
        )
        _shape = (self.frame_stack, screen_size, screen_size)
        self.observation_space = Box(
            low=_low, high=_high, shape=_shape, dtype=_obs_dtype
        )

        self.obs_to_stack = []
        self.current_obs_buffer_idx = 0

    @property
    def ale(self):
        """Make ale as a class property to avoid serialization error."""
        return self.env.unwrapped.ale

    def step(self, action):
        """Applies the preprocessing for an :meth:`env.step`."""
        total_reward, terminated, truncated, info = 0.0, False, False, {}

        for t in range(self.frame_skip):
            _, reward, terminated, truncated, info = self.env.step(action)

            total_reward += reward
            self.game_over = terminated

            if self.terminal_on_life_loss:
                new_lives = self.ale.lives()
                terminated = terminated or new_lives < self.lives
                self.game_over = terminated
                self.lives = new_lives

            if terminated or truncated:
                break

            self._buffer_frame()
            self._stack_obs()

        return np.stack(self.obs_to_stack, axis=0), total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Resets the environment using preprocessing."""
        # NoopReset
        self.obs_to_stack = []
        self.current_obs_buffer_idx = 0

        _, reset_info = self.env.reset(**kwargs)

        noops = (
            self.env.unwrapped.np_random.integers(1, self.noop_max + 1)
            if self.noop_max > 0
            else 0
        )

        noops = max(noops, self.frame_stack)
        for _ in range(noops):
            _, _, terminated, truncated, step_info = self.env.step(0)
            reset_info.update(step_info)
            if terminated or truncated:
                _, reset_info = self.env.reset(**kwargs)
            self._buffer_frame()
            self._stack_obs()

        self.lives = self.ale.lives()

        return np.stack(self.obs_to_stack, axis=0), reset_info

    def _stack_obs(self):
        obs = self._get_obs()
        self.obs_to_stack.append(obs)
        if len(self.obs_to_stack) > self.frame_stack:
            x = self.obs_to_stack.pop(0)
            del x

    def _buffer_frame(self):
        if self.grayscale_obs:
            self.ale.getScreenGrayscale(self.obs_buffer[self.current_obs_buffer_idx])
        else:
            self.ale.getScreenRGB(self.obs_buffer[self.current_obs_buffer_idx])

        self.current_obs_buffer_idx += 1
        if self.current_obs_buffer_idx == 2:
            self.current_obs_buffer_idx = 0

    def _get_obs(self):
        assert cv2 is not None
        np.maximum(self.obs_buffer[0], self.obs_buffer[1], out=self.frame_pool_buffer)
        obs = cv2.resize(
            self.frame_pool_buffer,
            (self.screen_size, self.screen_size),
            interpolation=cv2.INTER_AREA,
        )

        if self.scale_obs:
            obs = np.asarray(obs, dtype=np.float32) / 255.0
        else:
            obs = np.asarray(obs, dtype=np.uint8)

        if self.grayscale_obs and self.grayscale_newaxis:
            obs = np.expand_dims(obs, axis=0)  # Add a channel axis
        return obs
