import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # RL args
    parser.add_argument("--env_name", type=str)
    parser.add_argument(
        "--seed", default=0, type=int
    )  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--policy", type=str, default="TD3", choices=["TD3"])
    parser.add_argument(
        "--start_timesteps", default=25e3, type=int
    )  # Time steps initial random policy is used
    parser.add_argument(
        "--eval_freq", default=5e3, type=int
    )  # How often (time steps) we evaluate
    parser.add_argument("--eval_episodes", default=6, type=int)
    parser.add_argument(
        "--max_timesteps", default=1e6, type=int
    )  # Max time steps to run environment
    parser.add_argument(
        "--expl_noise", default=0.1, type=float
    )  # Std of Gaussian exploration noise
    parser.add_argument(
        "--batch_size", default=256, type=int
    )  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument(
        "--tau", default=0.005, type=float
    )  # Target network update rate
    parser.add_argument(
        "--policy_noise", default=0.2
    )  # Noise added to target policy during critic update
    parser.add_argument(
        "--noise_clip", default=0.5
    )  # Range to clip target policy noise
    parser.add_argument(
        "--policy_freq", default=2, type=int
    )  # Frequency of delayed policy updates
    parser.add_argument(
        "--save_model", action="store_true"
    )  # Save model and optimizer parameters
    parser.add_argument(
        "--load_model", default=""
    )  # Model load file name, "" doesn't load, "default" uses file_name

    # Metaworld args
    parser.add_argument(
        "--render_mode",
        type=str,
        default="rgb_array",
        choices=["rgb_array", "rgbd_array", "human"],
    )

    # Encoder args
    parser.add_argument(
        "--encoder", type=str, default="cnn", choices=["cnn", "resnet2d"]
    )
    parser.add_argument("--no_load_encoder", action="store_true")

    args = parser.parse_args()
    return args


def parse_test_args():
    parser = argparse.ArgumentParser()

    # RL args
    parser.add_argument("--env_name", type=str)
    parser.add_argument(
        "--seed", default=0, type=int
    )  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--policy", type=str, default="TD3", choices=["TD3"])
    parser.add_argument("--eval_episodes", default=6, type=int)
    parser.add_argument(
        "--load_model", default=""
    )  # Model load file name, "" doesn't load, "default" uses file_name

    # Metaworld args
    parser.add_argument(
        "--render_mode", type=str, default="rgb_array", choices=["rgb_array", "human"]
    )

    # Encoder args
    parser.add_argument("--encoder", type=str, default="cnn", choices=["cnn"])
    parser.add_argument("--no_load_encoder", action="store_true")

    # Testing args
    parser.add_argument("--ss_dir", type=str)
    parser.add_argument("--ss_freq", type=int, default=1)

    args = parser.parse_args()
    return args
