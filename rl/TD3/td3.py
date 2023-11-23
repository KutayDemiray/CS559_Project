import sys

sys.path.append("../../")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import copy

from typing import Tuple

from rl.replaybuffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    """TD3 policy network"""

    def __init__(
        self, state_dim: int, action_dim: int, action_lims: Tuple[float, float]
    ):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 256)
        self.l4 = nn.Linear(256, action_dim)

        self.action_min = action_lims[0]
        self.action_max = action_lims[1]

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = F.relu(self.l3(a))
        a = self.l4(a)

        # bounds tanh between action limits
        return ((self.action_max - self.action_min) / 2) * torch.tanh(a) + (
            self.action_max + self.action_min
        ) / 2


class Critic(nn.Module):
    """TD3 critic (contains two Q nets)"""

    def __init__(self, state_dim: int, action_dim: int):
        super(Critic, self).__init__()

        # Q1
        self.Q1_l1 = nn.Linear(state_dim + action_dim, 256)
        self.Q1_l2 = nn.Linear(256, 256)
        self.Q1_l3 = nn.Linear(256, 256)
        self.Q1_l4 = nn.Linear(256, 1)

        # Q2
        self.Q2_l1 = nn.Linear(state_dim + action_dim, 256)
        self.Q2_l2 = nn.Linear(256, 256)
        self.Q2_l3 = nn.Linear(256, 256)
        self.Q2_l4 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)

        q1 = F.relu(self.Q1_l1(sa))
        q1 = F.relu(self.Q1_l2(q1))
        q1 = F.relu(self.Q1_l3(q1))
        q1 = self.Q1_l4(q1)

        q2 = F.relu(self.Q2_l1(sa))
        q2 = F.relu(self.Q2_l2(q2))
        q2 = F.relu(self.Q2_l3(q2))
        q2 = self.Q2_l4(q2)

        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], dim=1)

        q1 = F.relu(self.Q1_l1(sa))
        q1 = F.relu(self.Q1_l2(q1))
        q1 = F.relu(self.Q1_l3(q1))
        q1 = self.Q1_l4(q1)

        return q1


class TD3:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_lims: Tuple[float, float],
        discount: float = 0.99,
        tau: float = 0.005,
        action_noise: float = 0.2,
        noise_clip: float = 0.5,
        update_freq: int = 2,
        lr: float = 3e-4,
    ):
        self.actor = Actor(state_dim, action_dim, action_lims).to(device)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr)

        self.action_min = action_lims[0]
        self.action_max = action_lims[1]
        self.discount = discount
        self.tau = tau  # update factor for target updates (lower tau retains more of target params compared to new ones)
        self.action_noise = action_noise  # i.e. mean of Gaussian noise (std is 1)
        self.noise_clip = noise_clip  # max noise (min will be its negative)
        self.update_freq = update_freq
        self.lr = lr

        self.iters = 0

    def get_action(self, state: np.ndarray):
        """Selects (noiseless) action given state"""
        state = torch.FloatTensor(state).reshape((1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train_batch(self, replay_buffer: ReplayBuffer, batch_size: int = 256):
        self.iters += 1

        # sample experience from buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # select action with policy and add clipped Gaussian noise
        with torch.no_grad():
            noise_vec = (torch.randn_like(action) * self.action_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_action = (self.actor_target(next_state) + noise_vec).clamp(
                self.action_min, self.action_max
            )

        # compute target Q value
        with torch.no_grad():
            # Q(s', a') from targets
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            # select minimum of two to reduce overestimation
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + not_done * self.discount * target_q

        # get current q values
        q1, q2 = self.critic(state, action)

        # compute loss to fit both q values towards target and optimize
        q1_loss = nn.MSELoss()
        q2_loss = nn.MSELoss()
        # q1.requires_grad = True
        # q2.requires_grad = True
        print(q1)
        print(q2)
        print(target_q)
        # target_q.requires_grad = True
        critic_loss = q1_loss(q1, target_q) + q2_loss(q2, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # periodically update policy and targets
        if self.iters % self.update_freq == 0:
            # optimize actor (maximize E[Q(s, policy(s)])
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            # update targets
            for param, target_param in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

            for param, target_param in zip(
                self.actor.parameters(), self.actor_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

    def save(self, filename: str):
        torch.save(self.critic.state_dict(), filename + "_critic.pt")
        torch.save(self.critic_opt.state_dict(), filename + "_critic_optimizer.pt")

        torch.save(self.actor.state_dict(), filename + "_actor.pt")
        torch.save(self.actor_opt.state_dict(), filename + "_actor_optimizer.pt")

    def load(self, filename: str):
        self.critic.load_state_dict(torch.load(filename + "_critic.pt"))
        self.critic_opt.load_state_dict(torch.load(filename + "_critic_optimizer.pt"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor.pt"))
        self.actor_opt.load_state_dict(torch.load(filename + "_actor_optimizer.pt"))
        self.actor_target = copy.deepcopy(self.actor)
