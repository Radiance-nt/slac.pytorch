try:
    from local_debug_logger import local_trace
except ImportError:
    local_trace = lambda: None
import gym
import cv2
import numpy
from gym.spaces import Box
from metaworld.benchmarks import ML1


gym.logger.set_level(40)


def make_dmc(domain_name, task_name, action_repeat, image_size=64,
             height=84,
             width=84,
             camera_id=0,
             frame_skip=1,
             episode_length=1000, ):
    assert domain_name == 'ML1'
    env = ML1.get_train_tasks(task_name)  # Create an environment with task `pick_place`
    tasks = env.sample_tasks(1)  # Sample a task (in this case, a goal variation)
    env.set_task(tasks[0])  # Set task
    max_episode_steps = (episode_length + frame_skip - 1) // frame_skip
    env = VisualizeWarpper(env, image_size, height, width)
    setattr(env, 'action_repeat', action_repeat)
    setattr(env, '_max_episode_steps', max_episode_steps)
    return env


class VisualizeWarpper(gym.Wrapper):
    def __init__(self, env,
                 image_size=64,
                 height=84,
                 width=84,
                 camera_id=0, ):
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
        image = self.env.get_image()
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = image.transpose(2, 0, 1)
        return image

    def step(self, action):
        result = self.env.step(action)
        image = self.env.get_image()
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = image.transpose(2, 0, 1)
        return (image,) + result[1:]

    def __getattr__(self, name):
        return getattr(self.env, name)
