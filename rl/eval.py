import metaworld
from rl.policy import Policy
import random
import torch
import os

import torch.nn as nn

from PIL import Image

from datetime import datetime

from train_util import *


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_metaworld_policy(
    policy: Policy,
    repr_net: nn.Module,
    env_name: str,
    seed: int = 42,
    eval_episodes: int = 10,
    render_mode: str = "rgbd_array",
    ss_dir: str = None,
    ss_freq: int = 1,
):
    """
    Runs policy for X episodes and returns average reward
    A fixed seed is used for the eval environment
    """
    # render_mode = "rgb_array"
    # eval_env.seed(seed + 100)
    ml1 = metaworld.ML1(env_name)
    eval_env = ml1.train_classes[env_name](
        render_mode=render_mode
    )  # Create an environment with task `window_open`
    task = random.choice(ml1.train_tasks)
    eval_env.set_task(task)  # Set task

    repr_net.eval()
    avg_reward = 0.0
    for ep in range(eval_episodes):
        if ss_dir is not None and not os.path.exists(os.path.join(ss_dir, f"{ep}")):
            os.mkdir(os.path.join(ss_dir, f"{ep}"))

        ss_t = 0

        state, done = eval_env.reset(seed=seed + 100), False
        state = state[0]

        state = observe(eval_env, repr_net, render_mode, train=False)

        while not done:
            action = policy.get_action(state)
            _, reward, terminated, truncated, _ = eval_env.step(action)

            if ss_dir is not None and ss_t % ss_freq == 0:
                state = observe(
                    eval_env,
                    repr_net,
                    render_mode,
                    train=False,
                    save_img_path=os.path.join(ss_dir, f"{ep}", f"{ss_t}.png"),
                )
            else:
                state = observe(eval_env, repr_net, render_mode, train=False)

            done = terminated or truncated
            avg_reward += reward

            ss_t += 1

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_metaworld_policy_perfectinfo(
    policy: Policy,
    repr_net,
    env_name: str,
    seed: int = 42,
    eval_episodes: int = 10,
    # render_mode: str = "rgb_array",
    ss_dir: str = None,
    ss_freq: int = 1,
):
    """
    Runs policy for X episodes and returns average reward
    A fixed seed is used for the eval environment
    """
    render_mode = "rgb_array"
    # eval_env.seed(seed + 100)
    ml1 = metaworld.ML1(env_name)
    eval_env = ml1.train_classes[env_name](
        render_mode=render_mode
    )  # Create an environment with task `window_open`
    task = random.choice(ml1.train_tasks)
    eval_env.set_task(task)  # Set task

    avg_reward = 0.0
    for ep in range(eval_episodes):
        if ss_dir is not None and not os.path.exists(os.path.join(ss_dir, f"{ep}")):
            os.mkdir(os.path.join(ss_dir, f"{ep}"))

        ss_t = 0

        state, done = eval_env.reset(seed=seed + 100), False
        state = state[0]
        state = torch.from_numpy(state)

        # rgb, seg, depth = eval_env.render()
        # rgb = torch.from_numpy(rgb.copy()).float().permute(2, 0, 1)
        print("eval", ep + 1)
        # state = repr_net(rgb)
        while not done:
            # print(state)
            action = policy.get_action(state.detach().numpy())
            next_state, reward, terminated, truncated, _ = eval_env.step(action)
            rgb, seg, depth = eval_env.render()

            if ss_dir is not None and ss_t % ss_freq == 0:
                rgb_im = Image.fromarray(rgb)
                rgb_im.save(os.path.join(ss_dir, f"{ep}", f"{ss_t}.png"))

            # rgb = torch.from_numpy(rgb.copy()).float().permute(2, 0, 1)
            # state = repr_net(rgb)
            done = terminated or truncated
            avg_reward += reward

            ss_t += 1

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward
