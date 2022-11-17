import argparse
import os
from datetime import datetime

import torch

from slac.algo import SlacAlgorithm
from slac.env import get_env
from slac.trainer import Trainer


def main(args):
    env, env_test = get_env(args)

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
        ckpt=args.ckpt
    )
    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        seed=args.seed,
        pretrain=not bool(args.ckpt)
    )
    trainer.train()
    print('Finished')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_steps", type=int, default=2 * 10 ** 6)
    parser.add_argument("--domain_name", type=str, default="cheetah")
    parser.add_argument("--task_name", type=str, default="run")
    parser.add_argument("--action_repeat", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None)

    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    main(args)
