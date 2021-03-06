import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
from rllab.core.serializable import Serializable


# class PendulumEnv(gym.Env):
#     metadata = {
#         'render.modes': ['human', 'rgb_array'],
#         'video.frames_per_second': 30
#     }
#
#     def __init__(self, g=10.0, color='red'):
#         self.max_speed = 8
#         self.max_torque = 2.
#         self.dt = .05
#         self.g = g
#         self.m = 1.
#         self.l = 1.
#         self.viewer = None
#         self.color = color
#         self._max_episode_steps = 200
#
#         high = np.array([1., 1., self.max_speed], dtype=np.float32)
#         self.action_space = spaces.Box(
#             low=-self.max_torque,
#             high=self.max_torque, shape=(1,),
#             dtype=np.float32
#         )
#         self.observation_space = spaces.Box(
#             low=-high,
#             high=high,
#             dtype=np.float32
#         )
#
#         self.seed()
#
#     def seed(self, seed=None):
#         self.np_random, seed = seeding.np_random(seed)
#         return [seed]
#
#     def step(self, u):
#         th, thdot = self.state  # th := theta
#
#         g = self.g
#         m = self.m
#         l = self.l
#         dt = self.dt
#
#         u = np.clip(u, -self.max_torque, self.max_torque)[0]
#         self.last_u = u  # for rendering
#         costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)
#         newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
#         newth = th + newthdot * dt
#         newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
#
#         self.state = np.array([newth, newthdot])
#         return self._get_obs(), -costs, False, {}
#
#     def reset(self, start=None):
#         if start is None:
#             high = np.array([np.pi, 1])
#             self.state = self.np_random.uniform(low=-high, high=high)
#         else:
#             self.state = np.array([start, 0])
#         self.last_u = None
#         return self._get_obs()
#
#     def _get_obs(self):
#         theta, thetadot = self.state
#         return np.array([np.cos(theta), np.sin(theta), thetadot])
#
#     def render(self, mode='human'):
#         if self.viewer is None:
#             from gym.envs.classic_control import rendering
#             self.viewer = rendering.Viewer(500, 500)
#             self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
#             rod = rendering.make_capsule(1, .2)
#             # rod.set_color(.8, .3, .3)
#             if self.color == 'red':
#                 rod.set_color(.8, .3, .3)
#             elif self.color == 'blue':
#                 rod.set_color(.288, .466, 1.)
#             else:
#                 print('pendulum color err!!!!')
#                 input()
#                 return
#             self.pole_transform = rendering.Transform()
#             rod.add_attr(self.pole_transform)
#             self.viewer.add_geom(rod)
#             axle = rendering.make_circle(.05)
#             axle.set_color(0, 0, 0)
#             self.viewer.add_geom(axle)
#             fname = "/home/wmingd/anaconda3/envs/GAIL/lib/python3.5/site-packages/gym/envs/classic_control/assets/clockwise.png"
#             # fname = path.join(path.dirname(__file__), "assets/clockwise.png")
#             self.img = rendering.Image(fname, 1., 1.)
#             self.imgtrans = rendering.Transform()
#             self.img.add_attr(self.imgtrans)
#
#         self.viewer.add_onetime(self.img)
#         self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
#         if self.last_u:
#             self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)
#
#         return self.viewer.render(return_rgb_array=(mode == 'rgb_array'))
#
#     def close(self):
#         if self.viewer:
#             self.viewer.close()
#             self.viewer = None


def angle_normalize(x):
    return (((x + np.pi) % (2 * np.pi)) - np.pi)


class GymPendulumEnv(gym.Env, Serializable):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }
    
    def __init__(self, color='red'):
        
        self.max_speed = 8
        self.max_torque = 2.
        self.dt = .05
        self.g = 10.0
        self.m = 1.
        self.l = 1.
        self.viewer = None
        self.color = color
        self._max_episode_steps = 200
        
        high = np.array([1., 1., self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque,
            high=self.max_torque, shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )
        
        self.seed()
        # "Serializable" seems important feature, from rl-lab
        Serializable.quick_init(self, locals())
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def step(self, a):
        th, thdot = self.state  # th := theta
        
        g = self.g
        m = self.m
        l = self.l
        dt = self.dt
        
        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)
        newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        newth = th + newthdot * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        
        self.state = np.array([newth, newthdot])
        return self._get_obs(), -costs, False, {}
    
    # is this called before each render?
    # or just called at the beginning?
    def get_viewer(self):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            # rod.set_color(.8, .3, .3)
            if self.color == 'red':
                rod.set_color(.8, .3, .3)
            elif self.color == 'blue':
                rod.set_color(.288, .466, 1.)
            else:
                print('pendulum color err!!!!')
                input()
                return
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = "/home/wmingd/anaconda3/envs/GAIL/lib/python3.5/site-packages/gym/envs/classic_control/assets/clockwise.png"
            # fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)
        
        return self.viewer
    
    def reset_mujoco(self, init_state=None):
        if start is None:
            high = np.array([np.pi, 1])
            self.state = self.np_random.uniform(low=-high, high=high)
        else:
            self.state = np.array([start, 0])
        self.last_u = None
        return self.get_current_obs()
    
    reset_trial = reset_mujoco
    
    def get_current_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])
    
    # equal to get_current_obs, just make it safer
    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])
    
    # i dont understand viewer_setup. It seems to be called before viewer.render()
    def viewer_setup(self):
        # !! not sure whether the following codes should be masked !!
        
        # if self.viewer is None:
        #     from gym.envs.classic_control import rendering
        #     self.viewer = rendering.Viewer(500, 500)
        #     self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
        #     rod = rendering.make_capsule(1, .2)
        #     # rod.set_color(.8, .3, .3)
        #     if self.color == 'red':
        #         rod.set_color(.8, .3, .3)
        #     elif self.color == 'blue':
        #         rod.set_color(.288, .466, 1.)
        #     else:
        #         print('pendulum color err!!!!')
        #         input()
        #         return
        #     self.pole_transform = rendering.Transform()
        #     rod.add_attr(self.pole_transform)
        #     self.viewer.add_geom(rod)
        #     axle = rendering.make_circle(.05)
        #     axle.set_color(0, 0, 0)
        #     self.viewer.add_geom(axle)
        #     fname = "/home/wmingd/anaconda3/envs/GAIL/lib/python3.5/site-packages/gym/envs/classic_control/assets/clockwise.png"
        #     # fname = path.join(path.dirname(__file__), "assets/clockwise.png")
        #     self.img = rendering.Image(fname, 1., 1.)
        #     self.imgtrans = rendering.Transform()
        #     self.img.add_attr(self.imgtrans)
        #
        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)
        pass

