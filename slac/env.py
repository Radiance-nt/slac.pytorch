import random
import cv2
import gym
import mujoco_py
import numpy
from gym.spaces import Box

gym.logger.set_level(40)


def get_env(args):
    if args.domain_name == "ML1":
        env = make_metaworld(domain_name=args.domain_name,
                             task_name=args.task_name,
                             seed=args.seed)
        env_test = make_metaworld(domain_name=args.domain_name,
                                  task_name=args.task_name,
                                  seed=args.seed)

        return env, env_test
    elif args.domain_name == "meld":

        env = make_meld()
        env_test = make_meld()

        return env, env_test
    else:
        env = make_dmc(
            domain_name=args.domain_name,
            task_name=args.task_name,
            action_repeat=args.action_repeat,
            image_size=64,
        )
        env_test = make_dmc(
            domain_name=args.domain_name,
            task_name=args.task_name,
            action_repeat=args.action_repeat,
            image_size=64,
        )
        return env, env_test


def make_metaworld(domain_name, task_name,
                   frame_skip=1,
                   episode_length=200,
                   fix_goal=True,
                   goal_observed=False,
                   seed=0):
    import metaworld
    from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
                                ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)
    assert domain_name == "ML1"
    if not fix_goal:
        ml1 = metaworld.ML1(task_name)  # Construct the benchmark, sampling tasks
        env = ml1.train_classes[task_name]()
        task = random.choice(ml1.train_tasks)
        env.set_task(task)  # Set task
    else:
        if goal_observed:
            env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[f"{task_name}-goal-observable"](seed=seed)
        else:
            env = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN[f"{task_name}-goal-hidden"](seed=seed)
    max_episode_steps = (episode_length + frame_skip - 1) // frame_skip
    env = StepLimitWarpper(env, max_episode_steps)
    env = VisualizeWarpper(env)
    env.action_repeat = 1
    return env


def make_meld():
    from .envs.cheetah.cheetah_vel import HalfCheetahVelEnv

    env = HalfCheetahVelEnv()
    # register all gym envs
    max_steps_dict = {"HalfCheetahVel-v0": 50, }
    env = StepLimitWarpper(env, max_steps_dict["HalfCheetahVel-v0"])
    env = VisualizeWarpper(env)
    env.action_repeat = 1
    return env


def make_dmc(domain_name, task_name, action_repeat, image_size=64):
    import dmc2gym
    env = dmc2gym.make(
        domain_name=domain_name,
        task_name=task_name,
        visualize_reward=False,
        from_pixels=True,
        height=image_size,
        width=image_size,
        frame_skip=action_repeat,
    )
    setattr(env, 'action_repeat', action_repeat)
    env.max_episode_steps = env._max_episode_steps
    return env


class GymWarpper(gym.Wrapper):
    def step(self, action):
        state, reward, done, trucated, info = self.env.step(action)
        return state, reward, done, info

    def seed(self, seed):
        pass


class StepLimitWarpper(gym.Wrapper):
    def __init__(self, env, limit=200):
        super(StepLimitWarpper, self).__init__(env)
        # self.limit = getattr(env, "_max_episode_steps") if hasattr(env, "_max_episode_steps") else limit
        self.limit = limit
        self.env.max_episode_steps = limit
        self.t = 0

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self.t += 1
        done = True if self.t == self.limit else done
        return state, reward, done, info

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self.t = 0
        return result


class VisualizeWarpper(gym.Wrapper):
    def __init__(self, env,
                 image_size=64,
                 height=84,
                 width=84,
                 camera_id=0, ):
        self.offscreen = mujoco_py.MjRenderContextOffscreen(env.sim, offscreen=True, opengl_backend='glfw')
        super(VisualizeWarpper, self).__init__(env)
        low = numpy.zeros((3, image_size, image_size), numpy.uint8)
        high = numpy.ones((3, image_size, image_size), numpy.uint8) * 255
        self.observation_space = Box(low, high)
        self.image_size = image_size
        self.height = height
        self.width = width
        self.camera_id = camera_id

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        image = self.get_image()
        return image

    def step(self, action):
        result = self.env.step(action)
        image = self.get_image()
        return (image,) + result[1:]

    def get_image(self):
        """comment /usr/lib/x86_64-linux-gnu/libGLEW.so"""
        self.offscreen.render(self.image_size, self.image_size, camera_id=0)
        image = self.offscreen.read_pixels(self.image_size, self.image_size, depth=False)

        # image = self.env.render('rgb_array')

        # image = self.env.sim.render(**{'height': self.image_size, 'width': self.image_size, 'device_id': 0})

        image = cv2.resize(image, (self.image_size, self.image_size))
        image = image.transpose(2, 0, 1)

        return image

    def __getattr__(self, name):
        return getattr(self.env, name)
