import numpy as np
import itertools
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding

import scipy.spatial
from matplotlib.path import Path

import logging

logger = logging.getLogger(__name__)


class RandomShapes(gym.Env):
    metadata = {'render.modes':['human', 'rgb_array']}

    def __init__(self, image_size=32, shape_complexity=10):
        self.image_size = image_size

        self._seed()
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(0, 1, [3, self.image_size, self.image_size])

        self._reset(shape_complexity=shape_complexity)

        self.viewer = None

    def _step(self, action):
        done = False
        actions = list(itertools.product([-1, 0, 1], repeat=2))
        actions.remove((0, 0))
        self.player_points.append(self.cursor)
        self.cursor = tuple(max(min(c + a, self.image_size - 1), 0) for c, a in zip(self.cursor, actions[action]))
        reward = 0

        if self.cursor in self.player_points:
            self.player_mask = self._get_polygon_mask(self.player_points)
            intersection = np.count_nonzero(self.player_mask * self.ground_truth)
            union = np.count_nonzero(self.player_mask) + np.count_nonzero(self.ground_truth) - intersection
            reward = intersection / union
            done = True

        cursor_mask = np.zeros_like(self.ground_truth)
        cursor_mask[self.cursor] = 1
        state = np.stack([self.image, self.player_mask, cursor_mask])
        return state, reward, done, {}

    def _reset(self, shape_complexity=10):
        # Create the polygon
        self.polygon_points = shape_complexity
        border_size = 2
        # Ground truth polygon
        points = np.random.randint(border_size, self.image_size - border_size,
            [self.polygon_points, 2])  # 30 random points in 2-D
        hull = scipy.spatial.ConvexHull(points)
        self.poly_verts = [(points[simplex, 0], points[simplex, 1]) for simplex in hull.vertices]

        # Ground truth pixel mask
        x, y = np.meshgrid(np.arange(self.image_size), np.arange(self.image_size))
        x, y = x.flatten(), y.flatten()
        path = Path(self.poly_verts)
        self.ground_truth = path.contains_points(np.vstack((x, y)).T).reshape([self.image_size, self.image_size])

        # Image
        self.image = self.ground_truth

        self.cursor = (self.poly_verts[0])
        self.player_points = [self.cursor]
        self.player_mask = np.zeros_like(self.ground_truth)
        cursor_mask = np.zeros_like(self.ground_truth)
        cursor_mask[self.cursor] = 1
        return np.stack([self.image, self.player_mask, cursor_mask])

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        scale_factor = 5
        scale_point = lambda p:tuple(scale_factor * c for c in p)
        scale_points = lambda points:[scale_point(p) for p in points]
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(scale_factor * self.image_size, scale_factor * self.image_size)
            # Ground truth
            ground_truth_polygon = rendering.FilledPolygon(scale_points(self.poly_verts))
            self.ground_truth_polygon_transformation = rendering.Transform()
            ground_truth_polygon.add_attr(self.ground_truth_polygon_transformation)
            ground_truth_polygon.set_color(0, 100, 0)
            self.viewer.add_geom(ground_truth_polygon)
            # Cursor
            cursor_polygon = rendering.make_circle(radius=scale_factor / 2)
            self.cursor_polygon_transformation = rendering.Transform()
            cursor_polygon.add_attr(self.cursor_polygon_transformation)
            cursor_polygon.set_color(100, 0, 0)
            self.viewer.add_geom(cursor_polygon)

        # Update for every frame
        self.cursor_polygon_transformation.set_translation(*scale_point(self.cursor))
        self.viewer.draw_polyline(scale_points(self.player_points))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    # def _close(self):
    #     super._close()
    #
    # def _configure(self, *args, **kwargs):
    #     super._configure(*args, **kwargs)

    def _seed(self, seed=None):
        np.random.seed(seed)

    def _get_polygon_mask(self, vertices):
        # Create vertex coordinates for each grid cell...
        # (<0,0> is at the top left of the grid in this system)
        x, y = np.meshgrid(np.arange(self.image_size), np.arange(self.image_size))
        x, y = x.flatten(), y.flatten()

        points = np.vstack((x, y)).T

        path = Path(vertices)
        grid = path.contains_points(points, radius=1)
        grid = grid.reshape([self.image_size, self.image_size])

        return grid


# A small testing interface for humans. Use numpad to move around
if __name__ == '__main__':
    import gym

    env = gym.envs.make('RandomShapes-v0')
    env.render()
    keymap = {'1':0, '4':1, '7':2, '2':3, '8':4, '3':5, '6':6, '9':7}
    while True:
        action = keymap.get(input())
        if action is None:
            print('Invalid action.')
            continue
        state, reward, done, _ = env.step(action)
        print(state.shape)
        print('Reward = {}, Done = {}'.format(reward, done))
        env.render()
        if done:
            env.reset()
            print('Enter anything to continue...')
            input()
