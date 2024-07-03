# Imports:
# --------
import os
import torch
import random
import numpy as np
from utils.dqn_utils import train_dqn, heatmap_evolution

 
# TODO: Add config and parser


##! Q-learning:
##! -----------
# TODO: Add Q-learning
# TODO: Convert Q-learning to GPU capable


##! DQN:
##! ----
# User definitions:
# -----------------
train_DQN = True
generate_DQN_heatmaps = True
dqn_exp_num = "4f"
results_path = "./results/DQN"
DEVICE = "cuda:0"
env_random_reset = False

# Hyperparameter definitions:
# ---------------------------
SAVING_FREQ = 100
TRAIN_PER_REPLAY_MEM = 5
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 800
TARGET_UPDATE_RATE = 0.005
LEARNING_RATE = 1e-3
REPLAY_MEM_MAX_SIZE = 50_000
NO_EPISODES = 10_000
MAX_STEPS_PER_EPISODE = 5_000

# Environment definitions:
# ------------------------
N_ACTIONS = 4
N_OBSERVATIONS = 2


##! PPO:
##! ----
# TODO: Add PPO code


##! SAC:
##! ----
# TODO: Add SAC code


# Run as script:
# --------------
if __name__ == "__main__":
    # Create a results folder if not present:
    # --------------------------------------
    if not os.path.isdir(results_path):
        os.mkdir(results_path)

    # Set seed value for repeatability:
    # ---------------------------------
    seed_no = 200

    torch.manual_seed(seed_no)
    random.seed(seed_no)
    np.random.seed(seed_no)

    # Train a DQN:
    # ------------
    if train_DQN:
        saving_path = os.path.join(results_path,
                                   f"DQN_{dqn_exp_num}")

        train_dqn(device=DEVICE,
                  saving_path=saving_path,
                  train_per_replay_mem=TRAIN_PER_REPLAY_MEM,
                  saving_freq=SAVING_FREQ,
                  batch_size=BATCH_SIZE,
                  num_episodes=NO_EPISODES,
                  max_episode_duration=MAX_STEPS_PER_EPISODE,
                  gamma=GAMMA,
                  max_replay_mem=REPLAY_MEM_MAX_SIZE,
                  n_observations=N_OBSERVATIONS,
                  n_actions=N_ACTIONS,
                  learning_rate=LEARNING_RATE,
                  target_update_rate=TARGET_UPDATE_RATE,
                  eps_start=EPS_START,
                  eps_end=EPS_END,
                  eps_decay=EPS_DECAY,
                  env_random_reset=env_random_reset)

    if generate_DQN_heatmaps:
        heatmap_evolution(freq=SAVING_FREQ,
                          model_path=os.path.join(
                              "results", "DQN", f"DQN_{dqn_exp_num}", "models"),
                          saving_path=os.path.join(
                              "results", "DQN", f"DQN_{dqn_exp_num}", "heatmaps"),
                          device=DEVICE)
