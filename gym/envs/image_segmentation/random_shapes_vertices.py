import gym
from gym.envs.image_segmentation.helper import *
import skimage.draw

import scipy.spatial
from matplotlib.path import Path

import logging

logger = logging.getLogger(__name__)


class RandomShapesVertices(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, image_size=32, shape_complexity=5):
        self.image_size = image_size
        self.shape_complexity = shape_complexity

        self._seed()

        self._reset()

        self.observation_space = gym.spaces.Box(0, 1, [self.image_size, self.image_size, 3])
        self.action_space = gym.spaces.Discrete(self.image_size ** 2)

        self.viewer = None

    def _step(self, action):
        done = False
        reward = 0

        # self.cursor = tuple(action)
        self.cursor = (action % self.image_size, action // self.image_size)

        # # Close the shape
        # if self.cursor == self.player_points[0]:
        #     player_polygon = np.zeros_like(self.image)
        #     rr, cc = skimage.draw.polygon(*zip(*self.player_points))
        #     player_polygon[rr, cc] = 1
        #     intersection = np.count_nonzero(player_polygon * self.ground_truth)
        #     union = np.count_nonzero(player_polygon) + np.count_nonzero(self.ground_truth) - intersection
        #     reward = intersection / union
        #     done = True

        # Self intersecting shape
        for i in range(1, len(self.player_points)):
            does_intersect, intersection = seg_intersect(np.array(self.player_points[i - 1]),
                                                         np.array(self.player_points[i]),
                                                         np.array(self.player_points[-1]), np.array(self.cursor))
            if does_intersect and not np.all(intersection == self.player_points[-1]):
                # Give reward
                player_polygon = np.zeros_like(self.image)
                rr, cc = skimage.draw.polygon(*zip(*self.player_points))
                player_polygon[rr, cc] = 1
                for i in range(0, len(self.player_points)):
                    rr, cc = skimage.draw.line(*self.player_points[i - 1], *self.player_points[i])
                    player_polygon[rr, cc] = 1
                intersection = np.count_nonzero(player_polygon * self.ground_truth)
                union = np.count_nonzero(player_polygon) + np.count_nonzero(self.ground_truth) - intersection
                reward = intersection / union if union != 0 else 0
                reward = reward ** 2
                #
                done = True
                break

        # Build the player mask
        self.player_points.append(self.cursor)
        for i in range(1, len(self.player_points)):
            rr, cc = skimage.draw.line(*self.player_points[i - 1], *self.player_points[i])
            self.player_mask[rr, cc] = 1

        # Build the cursor mask
        cursor_mask = np.zeros_like(self.ground_truth)
        cursor_mask[self.cursor] = 1

        state = np.stack([self.image, self.player_mask, cursor_mask], axis=2)
        return state, reward, done, {}

    def _reset(self):
        # Create the polygon
        self.polygon_points = self.shape_complexity

        # Ground truth polygon
        while True:
            try:
                points = np.random.randint(0, self.image_size, [self.polygon_points, 2])  # Random points in 2-D
                hull = scipy.spatial.ConvexHull(points)
                break
            except:
                # Keep trying until we have a valid polygon
                print('Hit error!')
                pass
        self.poly_verts = [(points[simplex, 0], points[simplex, 1]) for simplex in hull.vertices]

        # Ground truth pixel mask
        self.ground_truth = np.zeros([self.image_size, self.image_size])
        rr, cc = skimage.draw.polygon(*zip(*self.poly_verts))
        self.ground_truth[rr, cc] = 1
        for i in range(0, len(self.poly_verts)):
            rr, cc = skimage.draw.line(*self.poly_verts[i - 1], *self.poly_verts[i])
            self.ground_truth[rr, cc] = 1

        # Image
        self.image = self.ground_truth

        self.player_points = []
        self.player_mask = np.zeros_like(self.image)
        cursor_mask = np.zeros_like(self.image)
        self.cursor = None
        return np.stack([self.image, self.player_mask, cursor_mask], axis=2)

    # def _render(self, mode='human', close=False):
    #     from gym.envs.classic_control import rendering
    #     if close:
    #         if self.viewer is not None:
    #             self.viewer.close()
    #             self.viewer = None
    #         return
    #
    #     scale_factor = 10
    #     scale_point = lambda p: tuple(scale_factor * c + scale_factor / 2 for c in p)
    #     scale_points = lambda points: [scale_point(p) for p in points]
    #
    #     if self.viewer is None:
    #         self.viewer = rendering.Viewer(scale_factor * self.image_size, scale_factor * self.image_size)
    #         # Cursor
    #         cursor_polygon = rendering.make_circle(radius=scale_factor / 2)
    #         self.cursor_polygon_transformation = rendering.Transform()
    #         cursor_polygon.add_attr(self.cursor_polygon_transformation)
    #         cursor_polygon.set_color(100, 0, 0)
    #         self.viewer.add_geom(cursor_polygon)
    #
    #     # Ground truth
    #     ground_truth_polygon = rendering.FilledPolygon(scale_points(self.poly_verts))
    #     self.ground_truth_polygon_transformation = rendering.Transform()
    #     ground_truth_polygon.add_attr(self.ground_truth_polygon_transformation)
    #     ground_truth_polygon.set_color(0, 100, 0)
    #     self.viewer.add_onetime(ground_truth_polygon)
    #
    #     # Update for every frame
    #     if self.cursor is not None:
    #         self.cursor_polygon_transformation.set_translation(*scale_point(self.cursor))
    #     self.viewer.draw_polyline(scale_points(self.player_points))
    #
    #     return self.viewer.render(return_rgb_array=mode == 'rgb_array')

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


# A small testing interface for humans. Enter moves as tuples.
if __name__ == '__main__':
    import gym
    import matplotlib.pyplot as plt

    env = gym.envs.make('RandomShapesVertices-v0')
    state = env.reset()
    while True:
        print(env.poly_verts)
        plt.imshow(state, interpolation='nearest')
        plt.show(block=False)
        action = eval(input())
        plt.close()
        if action is None:
            print('Invalid action.')
            continue
        state, reward, done, _ = env.step(action)

        print(state.shape)
        print('Reward = {}, Done = {}'.format(reward, done))
        if done:
            plt.imshow(state, interpolation='nearest')
            plt.show(block=True)
            state = env.reset()
            print('Resetting...')

    env = gym.envs.make('RandomShapesVertices-v0')

    state = env.reset()
    rewards = []
    iterations = 100
    border = 16
    for _ in range(iterations):
        env.reset()
        state, reward, done, _ = env.step((border, border))
        state, reward, done, _ = env.step((border, env.image_size - 1 - border))
        state, reward, done, _ = env.step((
            env.image_size - 1 - border, env.image_size - 1 - border))
        state, reward, done, _ = env.step((env.image_size - 1 - border, border))
        state, reward, done, _ = env.step((border, border))
        rewards.append(reward)
        print('Reward = {}'.format(reward))
    print('avg reward={}'.format(sum(rewards) / iterations))

    ## image_size = 32
    # Border = 0 -> 0.46435
    # Border = 1 -> 0.52288
    # Border = 2 -> 0.5714
    # Border = 3 -> 0.6038
    # Border = 4 -> 0.6156
    # Border = 5 -> 0.60437
    # Border = 6 -> 0.56379
    # Border = 15-> 0.002323

    ## image_size = 128
    # Border = 0  -> 0.20713682604747558
    # Border = 4  -> 0.26284157985325285
    # Border = 8  -> 0.331898401189879
    # Border = 12 -> 0.3691057605767563
    # Border = 14 -> 0.3819274806397166
    # Border = 16 -> 0.3934860228735623
    # Border = 18 -> 0.3916663615541582
    # Border = 20 -> 0.366628872275845
    # Border = 24 -> 0.3500336330756317
    # Border = 32 -> 0.2213864452907915
