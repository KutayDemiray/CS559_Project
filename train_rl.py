import torch
import gymnasium as gym
import os
from rl.util import *
from parse_args import parse_args
import numpy as np

from rl.TD3.td3 import TD3
from rl.replaybuffer import ReplayBuffer

from encoder.smallcnn.smallcnn import SmallCNN
from torchsummary import summary

args = parse_args()
print(args)
file_name = f"{args.policy}_{args.env_name}_{args.seed}"
print("---------------------------------------")
print(f"Policy: {args.policy}, Env: {args.env_name}, Seed: {args.seed}")
print("---------------------------------------")

if not os.path.exists("./results"):
    os.makedirs("./results")

if args.save_model and not os.path.exists("./models"):
    os.makedirs("./models")


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
# print(env.action_space.shape)
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

# print("#############################", action_dim)

kwargs = {
    "state_dim": state_dim,
    "action_dim": action_dim,
    "action_lims": (-max_action, max_action),
    "discount": args.discount,
    "tau": args.tau,
}

# Initialize policy
if args.policy == "TD3":
    # Target policy smoothing is scaled wrt the action scale
    kwargs["action_noise"] = args.policy_noise * max_action
    kwargs["noise_clip"] = args.noise_clip * max_action
    kwargs["update_freq"] = args.policy_freq
    policy = TD3(**kwargs)

if args.load_model != "":
    policy_file = file_name if args.load_model == "default" else args.load_model
    policy.load(f"./models/{policy_file}")

replay_buffer = ReplayBuffer(state_dim, action_dim)

if args.no_load_encoder:
    repr_net = SmallCNN(state_dim)
else:
    pass

if args.env_name in [
    "drawer-open-v2",
    "soccer-v2",
    "window-open-v2",
    "hammer-v2",
]:
    # Evaluate untrained policy
    evaluations = [
        eval_metaworld_policy(
            policy, repr_net, args.env_name, seed=42, eval_episodes=args.eval_episodes
        )
    ]

state, done = env.reset(seed=42), False
state = state[0]
rgb, seg, depth = env.render()
rgb = torch.from_numpy(rgb.copy()).float().permute(2, 0, 1)
state = repr_net(rgb)
episode_reward = 0
episode_timesteps = 0
episode_num = 0

with torch.no_grad():
    for t in range(int(args.max_timesteps)):
        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                policy.get_action(state.detach().numpy())
                + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action
        _, reward, terminated, truncated, _ = env.step(action)

        rgb, seg, depth = env.render()
        rgb = torch.from_numpy(rgb.copy()).float().permute(2, 0, 1)
        next_state = repr_net(rgb)
        done = terminated or truncated
        # done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(
                f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}"
            )
            # Reset environment
            state, done = env.reset(), False
            state = state[0]
            rgb, seg, depth = env.render()
            rgb = torch.from_numpy(rgb.copy()).float().permute(2, 0, 1)
            state = repr_net(rgb)
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            print("eval")
            if args.env_name in [
                "drawer-open-v2",
                "soccer-v2",
                "window-open-v2",
                "hammer-v2",
            ]:
                evaluations.append(
                    eval_metaworld_policy(
                        policy,
                        repr_net,
                        args.env_name,
                        eval_episodes=args.eval_episodes,
                    )
                )
            if args.save_model:
                np.save(f"./results/{args.env_name}.pt", evaluations)
                if not os.path.exists(f"./bin/agent/{args.env_name}"):
                    os.mkdir(f"./bin/agent/{args.env_name}")
                policy.save(f"./bin/agent/{args.env_name}/{args.policy}_{args.encoder}")

print("done")
