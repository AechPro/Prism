from typing import Union
import gymnasium as gym
import numpy as np
from custom_environments.shapes import ShapesVisualizer

REMOVED = -1


class ShapesEnvironment(gym.Env):
    def __init__(self, starting_board: Union[str, np.ndarray], render_this_env=False):
        self.heights = None
        self.colors = None
        self._unique_moves = None
        self.board = None
        self.board_dimensions = (7, 9)
        self.n_colors = 4
        self.render_this_env = render_this_env

        self._build_board(starting_board)
        self._init_heights()

        self.visualizer = None
        self.ep_len = 0
        self.ep_len_limit = 50

        n_grid_cells = int(round(np.prod(self.board_dimensions)))
        self.action_space = gym.spaces.Discrete(n_grid_cells)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(n_grid_cells,), dtype=np.float32)
        self.base_reward = -1
        self.empty_click_reward = -10
        self.time_penalty = 1.05

    def _build_board(self, board):
        if isinstance(board, str):
            colors = sorted(set(c for c in board if not c.isspace()))
            board = board.strip().split("\n")
            board = [row.strip() for row in board]
            board = [[colors.index(block) for block in row] for row in board if row]

            # Transpose the board (but y tho?)
            self.board = np.asarray([[board[y][x] for y in range(len(board))][::-1] for x in range(len(board[0]))],
                                    dtype=np.float32)

            self.colors = tuple(colors)
        else:
            self.colors = tuple(set(block for column in board for block in column))
            self.board = board

    def _init_heights(self):
        heights = np.array([len(column) for column in self.board])
        for x, column in enumerate(self.board):
            self.board[x, :len(column)] = column
            heights[x] = len(column)
        self.heights = heights

    def reset(self, seed=None, return_info=True):
        self.ep_len = 0
        self._unique_moves = None

        rng = np.random.default_rng(seed)
        board = rng.integers(0, self.n_colors, self.board_dimensions, dtype=int)
        self._build_board(board)
        self._init_heights()

        if return_info:
            return self._build_obs(), {}

        return self._build_obs()

    def step(self, action):
        x = action % self.board.shape[0]
        y = action // self.board.shape[0]

        reward = 0
        if self.board[x, y] != REMOVED:
            self.remove_group(x, y)
            reward += self.base_reward * (self.time_penalty ** (self.ep_len+1))
        else:
            reward += self.empty_click_reward

        done = np.all(self.board == REMOVED)
        truncated = self.ep_len == self.ep_len_limit
        self.ep_len += 1

        if self.render_this_env:
            self.render()

        return self._build_obs(), reward, done, truncated, {}

    def render(self):
        if self.visualizer is None:
            self.visualizer = ShapesVisualizer(self.board.shape[0], self.board.shape[1], 50)

        self.visualizer.render(self.board[:, ::-1])

    def seed(self, seed):
        pass

    def close(self):
        if self.visualizer is not None:
            self.visualizer.close()

    def _build_obs(self):
        return self.board.ravel()

    @classmethod
    def random_game(cls, width: int, height: int, num_colors: int, seed=None):
        rng = np.random.default_rng(seed)
        board = rng.integers(0, num_colors, (width, height), dtype=int)
        return cls(board)

    @property
    def height(self):
        return np.max(self.heights)

    @property
    def is_done(self):
        return self.height == 0

    @property
    def width(self):
        return len(self.board)

    def count_unique_colors(self):
        return sum(np.any(self.board == color) for color in range(len(self.colors)))

    def __str__(self):
        def color_to_str(color):
            return str(self.colors[color])

        s = ""
        for y in reversed(range(self.height)):
            for x in range(self.width):
                if y < len(self.board[x]) and self.board[x][y] != REMOVED:
                    s += color_to_str(self.board[x][y])
                else:
                    s += " "
            s += "\n"
        return s

    def get_valid_moves(self):
        for x in range(self.width):
            for y in range(self.heights[x]):
                assert self.board[x, y] != REMOVED
                yield x, y

    def get_group(self, x: int, y: int):
        color = self.board[x][y]
        queue = [(x, y)]
        seen = np.zeros(self.board.shape, dtype=bool)
        while queue:
            x, y = queue.pop()
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.width
                        and 0 <= ny < len(self.board[nx])
                        and self.board[nx][ny] == color
                        and not seen[nx, ny]):
                    queue.append((nx, ny))
            seen[x, y] = True
        return seen

    def get_unique_moves(self):
        if self._unique_moves is not None:
            return self._unique_moves

        unique_moves = {}
        covered = np.zeros(self.board.shape, dtype=bool)
        for x, y in self.get_valid_moves():
            if covered[x, y]:
                continue
            group = self.get_group(x, y)
            unique_moves[(x, y)] = group
            covered |= group
        self._unique_moves = unique_moves
        return unique_moves

    def remove_group(self, x: int, y: int):
        group = self.get_group(x, y)
        self.board[group] = REMOVED
        # Move blocks down
        for x in range(self.width):
            column = self.board[x]
            column = column[column != REMOVED]
            self.board[x, :len(column)] = column
            self.board[x, len(column):] = REMOVED
            self.heights[x] = len(column)
        self._unique_moves = None  # Invalidate the cache
        return np.sum(group)

    def equivalent(self, *moves):
        x0, y0 = moves[0]
        group = self.get_group(x0, y0)
        for x, y in moves[1:]:
            if not group[x, y]:
                return False
        return True

    def __hash__(self):
        return hash(self.board.tobytes())

    def __eq__(self, other):
        return np.all(self.board == other.board)


def measure_sps():
    import time
    game = ShapesEnvironment.random_game(width=7, height=9, num_colors=4, seed=1234)

    num_ts = 100_000_000
    ts = 0
    obs, _ = game.reset()
    start_time = time.perf_counter()
    while ts < num_ts:

        obs, rew, done, truncated, _ = game.step(game.action_space.sample())
        if done or truncated:
            obs, _ = game.reset()

        game.render()
        time.sleep(1/60.)
        if ts > 0 and ts % 10000 == 0:
            time_passed = time.perf_counter() - start_time
            print("{} / {} | {:7.2f}".format(ts, num_ts, ts / time_passed))

        ts += 1


def eval_agent():
    from prism.config import SHAPES_ENV_CONFIG
    from prism.factory import agent_factory
    import torch
    import time
    checkpoint_dir = "../../data/checkpoints/shapes_environment/agent_checkpoint_36200000"

    config = SHAPES_ENV_CONFIG
    config.use_cuda_graph = False
    config.device = "cpu"
    game = ShapesEnvironment(starting_board="GPPOGOG\n"
                                            "OOOPOPB\n"
                                            "GBPBPOP\n"
                                            "PPGPGGB\n"
                                            "OPBOOBP\n"
                                            "POGOBOP\n"
                                            "OOPPBPP\n"
                                            "OBBOOPO\n"
                                            "PPPGOBB", render_this_env=True)

    obs_shape = game.observation_space.shape
    n_actions = game.action_space.n

    agent = agent_factory.build_agent(config, obs_shape, n_actions)
    agent.load(checkpoint_dir)
    # agent.eval()

    def _play_episode(_obs, _game):
        _n_steps = 0
        _done = False
        _truncated = False
        while not (_done or _truncated):
            agent_input = torch.as_tensor(_obs, dtype=torch.float32, device=config.device).view(1, -1)
            action = agent.forward(agent_input)
            _obs, rew, _done, _truncated, _ = _game.step(action)
            _n_steps += 1
            # time.sleep(1)
        return _n_steps, _done, _truncated

    solve_counts = []
    obs = game._build_obs()

    for ep_num in range(1000):
        n_steps, done, truncated = _play_episode(obs, game)
        if not truncated:
            solve_counts.append(n_steps)
            print("Solved board {} after {} steps".format(ep_num, n_steps))
            print(np.mean(solve_counts), np.std(solve_counts), np.min(solve_counts), np.max(solve_counts))
            print()
        else:
            print("!FAILED TO SOLVE BOARD!")

        obs, _ = game.reset(seed=ep_num)


if __name__ == "__main__":
    # measure_sps()
    eval_agent()
