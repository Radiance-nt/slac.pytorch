import random

import gym
import cv2
import numpy
from gym.spaces import Box
import metaworld
import mujoco_py


gym.logger.set_level(40)


def make_dmc(domain_name, task_name, action_repeat, image_size=64,
             height=84,
             width=84,
             camera_id=0,
             frame_skip=1,
             episode_length=200, ):
    assert domain_name == 'ML1'
    # env = ML1.get_train_tasks(task_name)  # Create an environment with task `pick_place`
    # tasks = env.sample_tasks(1)  # Sample a task (in this case, a goal variation)
    # env.set_task(tasks[0])  # Set task
    ml1 = metaworld.ML1(task_name)  # Construct the benchmark, sampling tasks
    env = ml1.train_classes[task_name]()
    task = random.choice(ml1.train_tasks)
    env.set_task(task)  # Set task
    max_episode_steps = (episode_length + frame_skip - 1) // frame_skip
    env = VisualizeWarpper(env, image_size, height, width)
    setattr(env, 'action_repeat', action_repeat)
    env = StepLimitWarpper(env, max_episode_steps)
    return env



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
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = image.transpose(2, 0, 1)
        return image

    def step(self, action):
        result = self.env.step(action)
        image = self.get_image()
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = image.transpose(2, 0, 1)
        return (image,) + result[1:]

    def get_image(self):
        self.offscreen.render(self.image_size , self.image_size , camera_id=-1)
        image = self.offscreen.read_pixels(self.image_size , self.image_size , depth=True)
        return image[0]

    def __getattr__(self, name):
        return getattr(self.env, name)


class StepLimitWarpper(gym.Wrapper):
    def __init__(self, env, limit = 200):
        self.env = env
        # self.limit = getattr(env, "_max_episode_steps") if hasattr(env, "_max_episode_steps") else limit
        self.limit =  limit
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