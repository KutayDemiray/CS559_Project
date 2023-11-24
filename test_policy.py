import metaworld
import gymnasium as gym
import random
import torch
import numpy as np
import os

from parse_args import parse_test_args

from rl.TD3.td3 import TD3
from rl.replaybuffer import ReplayBuffer
from rl.eval import eval_metaworld_policy

from encoder.smallcnn.smallcnn import SmallCNN

args = parse_test_args()
print(args)

file_name = f"{args.policy}_{args.env_name}_{args.seed}"
print("---------------------------------------")
print(f"Policy: {args.policy}, Env: {args.env_name}, Seed: {args.seed}")
print("---------------------------------------")

if not os.path.exists("./results"):
    os.makedirs("./results")

# Initialize environment
if args.env_name in [
    "drawer-open-v2",
    "soccer-v2",
    "window-open-v2",
    "hammer-v2",
]:  # metaworld envs
    ml1 = metaworld.ML1(args.env_name)  # Construct the benchmark, sampling tasks

    env = ml1.train_classes[args.env_name](
        render_mode=args.render_mode
    )  # Create an environment with task `window_open`
    task = random.choice(ml1.train_tasks)
    env.set_task(task)  # Set task
elif args.env_name == "LunarLander-v2":  # special case gymnasium
    env = gym.make(args.env_name, continuous=True)
else:  # other gymnasium
    env = gym.make(args.env_name)

# Set seeds
# env.seed(args.seed)
env.action_space.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

kwargs = {
    "state_dim": state_dim,
    "action_dim": action_dim,
    "action_lims": (-max_action, max_action),
    #    "discount": args.discount,
    #    "tau": args.tau,
}

# Initialize policy
if args.policy == "TD3":
    # Target policy smoothing is scaled wrt the action scale
    #    kwargs["action_noise"] = args.policy_noise * max_action
    #    kwargs["noise_clip"] = args.noise_clip * max_action
    #    kwargs["update_freq"] = args.policy_freq
    policy = TD3(**kwargs)


if args.load_model != "":
    policy_file = file_name if args.load_model == "default" else args.load_model
    policy.load(f"{policy_file}")

replay_buffer = ReplayBuffer(state_dim, action_dim)

if args.no_load_encoder:
    repr_net = SmallCNN(state_dim)
else:
    pass

if not os.path.exists(args.ss_dir):
    os.mkdir(args.ss_dir)

# Evaluate untrained policy
evaluations = [
    eval_metaworld_policy(
        policy,
        repr_net,
        args.env_name,
        args.seed,
        eval_episodes=args.eval_episodes,
        ss_dir=args.ss_dir,
        ss_freq=args.ss_freq
        # render_mode="human",
    )
]
