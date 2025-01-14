import argparse
import os
from datetime import datetime

from slac.env import make_dmc
from slac.trainer import Trainer
from slac.algo import SlacAlgorithm

import torch


def main(args):
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

    log_dir = os.path.join(
        "log",
        f"{args.domain_name}-{args.task_name}",
        f'slac-seed{args.seed}-{datetime.now().strftime("%Y%m%d-%H%M")}',
    )

    algo = SlacAlgorithm(
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        action_repeat=args.action_repeat,
        device=torch.device("cuda" if args.cuda else "cpu"),
        seed=args.seed,
    )
    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        seed=args.seed,
        num_steps=2 * 10 ** 6,
        initial_collection_steps=10 ** 4,
        initial_learning_steps=10 ** 5,
    )
    trainer.train()
    print('Finished')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_steps", type=int, default=2 * 10 ** 6)
    parser.add_argument("--domain_name", type=str, default="ML1")
    parser.add_argument("--task_name", type=str, default="reach-v2")
    parser.add_argument("--action_repeat", type=int, default=4)
    parser.add_argument("--num_steps", type=int, default=2 * 10 ** 6)
    parser.add_argument("--initial_collection_steps", type=int, default= 10 ** 4,)
    parser.add_argument("--initial_learning_steps", type=int, default=10 ** 5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    main(args)
