
#! Paper: 
# Critical States in Reinforcement Learning: A Survey and New Perspectives

#! Institute:
# AImotion Bavaria, Technische Hochschule Ingolstadt (THI), Esplanade 10, 85049, Ingolstadt, Germany

# The project uses three different Reinforcement Learning algorithms.

#! Code and usage:
# The DQN code is adpated from the official Pytorch tutorial for CartPole (URL: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)

# The PPO code is adapted from MinimalRL GitHub repository (URL: https://github.com/seungeunrho/minimalRL/blob/master/ppo.py). MinialRL uses an MIT license.

# The SAC code is adapated from the following GitHub repository (URL: https://github.com/tsmatz/reinforcement-learning-tutorials/blob/master/06-sac.ipynb
# ) by Mr. Tsuyoshi Matsuzaki. There is no license mentioned for this repository. An attempt has been made to contact the author Mr. Tsuyoshi Matsuzaki via LinkedIn on 04.07.2024, 14:15.
#! Response: No response yet as of 04.07.2024

#! Declaration:
# The usage of this code is performed in good faith for academic research purposes only.



# Imports:
# --------
import os
import torch
import random
import numpy as np
from utils.dqn_utils import train_dqn, heatmap_evolution_dqn
from utils.sac_utils import train_sac, heatmap_evolution_sac


## General settings:
## -----------------
RENDER = False
DEVICE = "cuda:0"
FLATLAND_GAP = 0.15
FLATLAND_N_ACTIONS = 4
ACTION_STEP_SIZE = 0.5
FLATLAND_N_OBSERVATIONS = 2


# TODO: Add config and parser


##! Q-learning:
##! -----------
# TODO: Add Q-learning
# TODO: Convert Q-learning to GPU capable


##! DQN:
##! ----
# User definitions:
# -----------------
DQN_TRAIN = False
DQN_GENERATE_HEATMAPS = False
DQN_EXP_NUM = "paper1_results"
DQN_RESULTS_PATH = "./results/DQN"
DQN_ENV_RANDOM_RESET = False
DQN_SAVING_FREQ = 100

# Hyperparameter definitions:
# ---------------------------
DQN_TRAIN_PER_REPLAY_MEM = 5
DQN_BATCH_SIZE = 128
DQN_GAMMA = 0.99

DQN_EPS_START = 0.9
DQN_EPS_END = 0.01
DQN_EPS_DECAY = 300

DQN_TARGET_UPDATE_RATE = 0.005
DQN_LEARNING_RATE = 1e-3
DQN_REPLAY_MEM_MAX_SIZE = 50_000
DQN_NO_EPISODES = 2_000
DQN_MAX_STEPS_PER_EPISODE = 5_000


##! PPO:
##! ----
# TODO: Add PPO code


##! SAC:
##! ----
# User definitions:
# -----------------
SAC_TRAIN = False
SAC_GENERATE_HEATMAPS = True
SAC_EXP_NUM = "paper1_results"
SAC_SAVE_FREQ = 100
SAC_RESULTS_PATH = "./results/SAC"
SAC_ENV_RANDOM_RESET = True

SAC_MAX_BUFFER_SIZE = 100_000
SAC_MAX_STEPS = 3_000
SAC_NO_EPISODES = 2_000
SAC_BATCH_SIZE = 250
SAC_ALPHA = 0.1
SAC_LEARNING_RATE = 3*1e-5
SAC_GAMMA = 0.95
SAC_TAU = 0.002


# Run as script:
# --------------
if __name__ == "__main__":
    # Create a results folder if not present:
    # --------------------------------------
    if not os.path.isdir("./results"):
        os.mkdir("./results")


    #! DQN:
    #! ----
    # Set seed value for repeatability:
    # ---------------------------------
    seed_no = 200

    torch.manual_seed(seed_no)
    random.seed(seed_no)
    np.random.seed(seed_no)

    if not os.path.isdir(DQN_RESULTS_PATH):
        os.mkdir(DQN_RESULTS_PATH)

    saving_path = os.path.join(DQN_RESULTS_PATH,
                                f"DQN_{DQN_EXP_NUM}")

    # Train a DQN:
    # ------------
    if DQN_TRAIN:
        train_dqn(device=DEVICE,
                  render=RENDER,
                  saving_path=saving_path,
                  train_per_replay_mem=DQN_TRAIN_PER_REPLAY_MEM,
                  saving_freq=DQN_SAVING_FREQ,
                  batch_size=DQN_BATCH_SIZE,
                  num_episodes=DQN_NO_EPISODES,
                  max_episode_duration=DQN_MAX_STEPS_PER_EPISODE,
                  gamma=DQN_GAMMA,
                  max_replay_mem=DQN_REPLAY_MEM_MAX_SIZE,
                  n_observations=FLATLAND_N_OBSERVATIONS,
                  n_actions=FLATLAND_N_ACTIONS,
                  learning_rate=DQN_LEARNING_RATE,
                  target_update_rate=DQN_TARGET_UPDATE_RATE,
                  eps_start=DQN_EPS_START,
                  eps_end=DQN_EPS_END,
                  eps_decay=DQN_EPS_DECAY,
                  env_random_reset=DQN_ENV_RANDOM_RESET,
                  env_gap_size=FLATLAND_GAP,
                  action_step_size=ACTION_STEP_SIZE)


    if DQN_GENERATE_HEATMAPS:
        heatmap_evolution_dqn(freq=DQN_SAVING_FREQ,
                          model_path=os.path.join(
                              "results", "DQN", f"DQN_{DQN_EXP_NUM}", "models"),
                          saving_path=os.path.join(
                              "results", "DQN", f"DQN_{DQN_EXP_NUM}", "heatmaps"),
                          device=DEVICE)
    
    #! SAC:
    #! ----
    # Set seed value for repeatability:
    # ---------------------------------
    seed_no = 100

    torch.manual_seed(seed_no)
    random.seed(seed_no)
    np.random.seed(seed_no)

    if not os.path.isdir(SAC_RESULTS_PATH):
        os.mkdir(SAC_RESULTS_PATH)

    saving_path = os.path.join(SAC_RESULTS_PATH, 
                                f"SAC_{SAC_EXP_NUM}")
    
    # Train SAC:
    # ----------
    if SAC_TRAIN:
        train_sac(no_episodes=SAC_NO_EPISODES,
                  max_buffer_size=SAC_MAX_BUFFER_SIZE,
                  max_steps=SAC_MAX_STEPS,
                  batch_size=SAC_BATCH_SIZE,
                  alpha=SAC_ALPHA,
                  learning_rate=SAC_LEARNING_RATE,
                  gamma=SAC_GAMMA,
                  tau=SAC_TAU,
                  render=RENDER,
                  save_interval=SAC_SAVE_FREQ,
                  env_gap_size=FLATLAND_GAP,
                  env_random_reset=SAC_ENV_RANDOM_RESET,
                  device=DEVICE,
                  save_path=saving_path)

    # Generate SAC heatmaps:
    # ----------------------
    if SAC_GENERATE_HEATMAPS:
        heatmap_evolution_sac(device=DEVICE, 
                              model_path=os.path.join(saving_path, "models"), 
                              saving_path=saving_path, 
                              batch_size=SAC_BATCH_SIZE, 
                              freq=SAC_SAVE_FREQ)  
