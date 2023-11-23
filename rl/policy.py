from rl.replaybuffer import ReplayBuffer


class Policy:
    def get_action(self, obs):
        pass

    def train(self, replay_buffer: ReplayBuffer, batch_size: int):
        pass

    def save(self, filename: str):
        pass

    def load(self, filename: str):
        pass
