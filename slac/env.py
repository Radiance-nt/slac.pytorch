import dmc2gym
import gym

gym.logger.set_level(40)


def make_dmc(domain_name, task_name, action_repeat, image_size=64):
    env = dmc2gym.make(
        domain_name=domain_name,
        task_name=task_name,
        visualize_reward=False,
        from_pixels=False,
        height=image_size,
        width=image_size,
        frame_skip=action_repeat,
    )
    setattr(env, 'action_repeat', action_repeat)
    env = GymWarpper(env)
    return env


class GymWarpper(gym.Wrapper):
    def step(self, action):
        state, reward, done, trucated, info = self.env.step(action)
        return state, reward, done, info

    def seed(self, seed):
        pass
