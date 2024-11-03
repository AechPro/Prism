import numpy as np
from .tile_map import TileMap
import gymnasium as gym


class SimpleTrapEnvironment(gym.Env):
    def __init__(self, node_map=None, seed=123, render_this_env=False):
        self.rng = np.random.RandomState(seed)
        self.visualizer = None
        self.render_this_env = render_this_env

        if node_map is None:
            from .maps import SimpleTrapMap
            node_map = SimpleTrapMap()

        self.map = TileMap(node_map=node_map, rng=self.rng)

        self.unwalkable_coords = []
        for i in range(len(self.map.nodes)):
            for j in range(len(self.map.nodes[i])):
                if not self.map.nodes[i][j].walkable:
                    self.unwalkable_coords.append((self.map.nodes[i][j].x, self.map.nodes[i][j].y))

        self.current_node = None
        self.episode_length = 115
        self.current_step = 0
        self.MAX_X = int(round(self.map.width * self.map.node_radius))
        self.MAX_Y = int(round(self.map.height * self.map.node_radius))
        # self.spawn_point = (self.MAX_X//2, self.MAX_Y//2)
        # self.current_node = self.map.get_node(*spawn_point)

        spawn_points = []
        for x in range(3):
            for y in range(3):
                map_x = self.map.width //2 - 1 + x
                map_y = self.map.height // 2 - 1 + y
                node_x = map_x * self.map.node_radius
                node_y = map_y * self.map.node_radius
                spawn_points.append((node_x, node_y))
        self.spawn_points = spawn_points
        spawn_point = self.spawn_points[self.rng.randint(len(self.spawn_points))]
        self.current_node = self.map.get_node(*spawn_point)

        self.middle_x = self.current_node.x

        self.action_space = gym.spaces.Discrete(9)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)

    def reset(self, seed=None, return_info=True):
        if seed is not None:
            self._set_seed(seed)

        spawn_point = self.spawn_points[self.rng.randint(len(self.spawn_points))]
        # spawn_point = self.spawn_point
        self.current_node = self.map.get_node(*spawn_point)
        self.current_step = 0

        return self._form_obs(), {}

    def step(self, action):
        prev_x = self.current_node.x
        self.current_node = self.current_node.take_action(action)
        self.current_node.visit_count += 1
        x_dist = self.current_node.x - prev_x
        done = self.current_step >= self.episode_length

        reward = x_dist / self.map.node_radius

        self.current_step += 1

        obs = self._form_obs()
        if self.render_this_env:
            self.render()

        return obs.flatten(), reward, done, False, {}

    def _form_obs(self):
        x = self.current_node.x
        y = self.current_node.y
        obs = [x/self.MAX_X, y/self.MAX_Y]

        n_detected = 0
        left_idx = 0
        top_left_idx = 1
        up_idx = 2
        top_right_idx = 3
        right_idx = 4
        bot_right_idx = 5
        down_idx = 6
        bot_left_idx = 7

        dists = [-1 for _ in range(8)]

        for coords in self.unwalkable_coords:
            nx, ny = coords
            y_diff = abs(ny - y)
            x_diff = abs(nx - x)
            if ny == y:
                if nx > x:
                    dists[right_idx] = x_diff / self.MAX_X
                else:
                    dists[left_idx] = x_diff / self.MAX_X
                n_detected += 1

            elif nx == x:
                if ny > y:
                    dists[up_idx] = y_diff / self.MAX_Y
                else:
                    dists[down_idx] = y_diff / self.MAX_Y
                n_detected += 1

            elif y_diff == x_diff:
                d = 1.4142 * y_diff

                if nx > x and ny > y:
                    dists[top_right_idx] = d / self.MAX_X
                elif nx > x and ny < y:
                    dists[bot_right_idx] = d / self.MAX_X
                elif nx < x and ny > y:
                    dists[top_left_idx] = d / self.MAX_X
                elif nx < x and ny < y:
                    dists[bot_left_idx] = d / self.MAX_X

                n_detected += 1
            if n_detected == 8:
                break

        obs += dists
        return np.asarray(obs)

    def close(self):
        if self.visualizer is not None:
            self.visualizer.close()

        if self.map is not None:
            self.map.cleanup()

    def render(self, mode='human'):
        if self.visualizer is None:
            from . import Visualizer
            self.visualizer = Visualizer(self.map)

        self.current_node.debug = True
        self.visualizer.render()
        self.current_node.debug = False

    def _set_seed(self, seed):
        self.rng.seed(seed)
