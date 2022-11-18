import logging
import os
from collections import deque
from datetime import timedelta
from time import sleep, time

import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class SlacObservation:
    """
    Observation for SLAC.
    """

    def __init__(self, state_shape, action_shape, num_sequences):
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.num_sequences = num_sequences

    def reset_episode(self, state):
        self._state = deque(maxlen=self.num_sequences)
        self._action = deque(maxlen=self.num_sequences - 1)
        for _ in range(self.num_sequences - 1):
            self._state.append(np.zeros(self.state_shape, dtype=np.uint8))
            self._action.append(np.zeros(self.action_shape, dtype=np.float32))
        self._state.append(state)

    def append(self, state, action):
        self._state.append(state)
        self._action.append(action)

    @property
    def state(self):
        return np.array(self._state)[None, ...]

    @property
    def action(self):
        return np.array(self._action).reshape(1, -1)


class MeldTrainer:
    """
    Trainer for SLAC.
    """

    def __init__(
            self,
            env,
            env_test,
            algo,
            log_dir,
            seed=0,
            num_steps=3 * 10 ** 6,
            initial_collection_steps=10 ** 4,
            initial_learning_steps=10 ** 5,
            num_sequences=8,
            eval_interval=10 ** 4,
            num_eval_episodes=5,
            pretrain=True,
    ):
        # Env to collect samples.
        self.env = env
        self.env.seed(seed)

        # Env for evaluation.
        self.env_test = env_test
        self.env_test.seed(2 ** 31 - seed)

        # Observations for training and evaluation.
        self.ob = SlacObservation(env.observation_space.shape, env.action_space.shape, num_sequences)
        self.ob_test = SlacObservation(env.observation_space.shape, env.action_space.shape, num_sequences)

        # Algorithm to learn.
        self.algo = algo

        # Log setting.
        self.log = {"step": [], "return": []}
        self.csv_path = os.path.join(log_dir, "log.csv")
        self.log_dir = log_dir
        self.summary_dir = os.path.join(log_dir, "summary")
        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.model_dir = os.path.join(log_dir, "model")
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # Other parameters.
        self.action_repeat = self.env.action_repeat
        self.num_steps = num_steps
        self.initial_collection_steps = initial_collection_steps
        self.initial_learning_steps = initial_learning_steps
        self.eval_interval = eval_interval
        self.num_eval_episodes = num_eval_episodes
        self.pretrain = pretrain

    def train(self):
        max_episode_len = 50

        # ==============================================================================
        # Tasks and eval
        # ==============================================================================

        episodes_per_trial = 1
        num_train_tasks = 20
        eval_interval = 100

        # ==============================================================================
        # Initial collection + training
        # ==============================================================================

        init_collect_trials_per_task = 20

        init_model_train_ratio = 0.3

        # ==============================================================================
        # Data collection
        # ==============================================================================

        replay_buffer_capacity = 10000
        num_tasks_to_collect_per_iter = 1
        collect_trials_per_task = 1

        # ==============================================================================
        # Sample data for training
        # ==============================================================================

        num_tasks_per_train = 10
        train_trials_per_task = 1

        model_bs_in_steps = 256
        ac_bs_in_steps = 256

        # ==============================================================================
        # Training
        # ==============================================================================

        model_train_ratio = 0.8
        ac_train_ratio = 0.8

        model_train_freq = 1
        ac_train_freq = 1

        # ==============================================================================
        # General
        # ==============================================================================

        num_iterations = 10000000
        seed = 0

        # convert to number of steps
        env_steps_per_trial = episodes_per_trial * max_episode_len
        real_env_steps_per_trial = episodes_per_trial * (max_episode_len + 1)
        env_steps_per_iter = num_tasks_to_collect_per_iter * collect_trials_per_task * env_steps_per_trial
        per_task_collect_steps = collect_trials_per_task * env_steps_per_trial

        # initial collect + train
        init_collect_env_steps = num_train_tasks * init_collect_trials_per_task * env_steps_per_trial
        init_model_train_steps = int(init_collect_env_steps * init_model_train_ratio)

        # collect + train
        collect_env_steps_per_iter = num_tasks_to_collect_per_iter * per_task_collect_steps
        model_train_steps_per_iter = int(env_steps_per_iter * model_train_ratio)
        ac_train_steps_per_iter = int(env_steps_per_iter * ac_train_ratio)
        stop_model_training = 1E10

        # Time to start training.
        self.start_time = time()
        # Episode's timestep.
        t = 0
        # Initialize the environment.
        state = self.env.reset()
        self.ob.reset_episode(state)
        self.algo.buffer.reset_episode(state)

        train_tasks = self.env.init_tasks(num_tasks=num_train_tasks, is_eval_env=False)
        list_of_collect_task_idxs = np.random.choice(len(train_tasks), num_tasks_to_collect_per_iter,
                                                     replace=False)

        # pretrain
        for task_idx in range(num_train_tasks):
            self.env.set_task_for_env(train_tasks[task_idx])
            for count in range(init_collect_trials_per_task):
                t = self.algo.step(self.env, self.ob, t, True)

        if self.pretrain:
            # Update latent variable model first so that SLAC can learn well using (learned) latent dynamics.
            bar = tqdm(range(init_model_train_steps))
            for _ in bar:
                bar.set_description("Updating latent variable model.")
                self.algo.update_latent(self.writer)
            self.algo.save_model(os.path.join(self.model_dir, f"pretrain_latent"))


        for iteration in range(num_iterations):
            for count, task_idx in enumerate(list_of_collect_task_idxs):
                # set task for the env
                self.env.set_task_for_env(train_tasks[task_idx])

                # collect data with collect policy
                t = self.algo.step(self.env, self.ob, t, False)

            ####################
            # train model
            ####################
            if (iteration == 0) or (
                    (iteration % model_train_freq == 0) and (t < stop_model_training)):
                logging.info('\n\nPerforming %d steps of model training, each on %d random tasks',
                             model_train_steps_per_iter, num_tasks_per_train)
                for model_iter in range(model_train_steps_per_iter):
                    # train model
                    self.algo.update_latent(self.writer)

            ####################
            # train actor critic
            ####################
            if iteration % ac_train_freq == 0:
                logging.info('\n\nPerforming %d steps of AC training, each on %d random tasks \n\n',
                             ac_train_steps_per_iter, num_tasks_per_train)
            for ac_iter in range(ac_train_steps_per_iter):
                # train ac
                self.algo.update_sac(self.writer)

            if eval_interval and (iteration % eval_interval == 0):
                self.evaluate(iteration)
                self.algo.save_model(os.path.join(self.model_dir, f"step{0}"))

    def evaluate(self, step_env):
        mean_return = 0.0
        mean_success = 0.0

        num_eval_tasks = 10

        eval_interval = 100
        num_eval_trials = 10

        eval_tasks = self.env.init_tasks(num_tasks=num_eval_tasks, is_eval_env=False)

        for task_idx in range(num_eval_tasks):
            self.env.set_task_for_env(eval_tasks[task_idx])
            for i in range(self.num_eval_episodes):
                state = self.env_test.reset()
                self.ob_test.reset_episode(state)
                episode_return = 0.0
                done = False

                while not done:
                    action = self.algo.exploit(self.ob_test)
                    state, reward, done, info = self.env_test.step(action)
                    self.ob_test.append(state, action)
                    episode_return += reward

                mean_return += episode_return / self.num_eval_episodes
                # mean_success += (info.get('success') or 0) / self.num_eval_episodes

        # Log to CSV.
        self.log["step"].append(step_env)
        self.log["return"].append(mean_return)
        pd.DataFrame(self.log).to_csv(self.csv_path, index=False)

        # Log to TensorBoard.
        self.writer.add_scalar("return/test", mean_return, step_env)
        self.writer.add_scalar("success/test", mean_success, step_env)
        print(f"Steps: {step_env:<6}   " f"Return: {mean_return:<5.1f}   " f"Time: {self.time}")

    @property
    def time(self):
        return str(timedelta(seconds=int(time() - self.start_time)))
