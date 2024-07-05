# TODO: Change PPO to the required format

# Imports:
# --------
import os
import math
import torch
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.distributions import Categorical
from src.environments.continuous.flatland import ContinuousFlatLand

# NOTE: Set a seed value for reproducibility
# -----
seed_value = 100

torch.manual_seed(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

# Hyperparameters:
# ----------------
learning_rate = 3*1e-4
gamma = 0.99
lmbda = 0.85
eps_clip = 0.2
K_epoch = 5
num_steps = 3_000
num_episodes = 10_000
save_interval = 100
render_interval = 100
gap_size = 0.15

#! We shouldn't use entropy decay
entropy = 0.1
# entropy_coeff_start = 0.1
# entropy_coeff_end = 0.01
# decay_rate = 1500

state_dim = 2
action_dim = 4

print(f"learning_rate: {learning_rate}, lambda: {lmbda}")


# Device configuration
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


class PPO(nn.Module):
    def __init__(self, state_dim=2, action_dim=4, learning_rate=learning_rate):
        super(PPO, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)

        self.fc_pi = nn.Linear(64, action_dim)
        self.fc_v = nn.Linear(64, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.to(device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x

    def pi(self, x, softmax_dim=-1):
        x = self.forward(x)
        x = self.fc_pi(x)

        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = self.forward(x)

        val = self.fc_v(x)
        return val

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        self.data = np.array(self.data, dtype=np.float32)

        self.s = torch.from_numpy(self.data[:, 0, :]).to(device)
        self.a = torch.from_numpy(np.array(self.data[:, 1, 0:1],
                                           dtype=np.int64)).to(device)

        self.r = torch.from_numpy(self.data[:, 2, 0:1]).to(device)
        self.s_prime = torch.from_numpy(self.data[:, 3, :]).to(device)
        self.prob_a = torch.from_numpy(self.data[:, 4, 0:1]).to(device)
        self.done = torch.from_numpy(1-self.data[:, 5, 0:1]).to(device)

        self.data = []

    def train_net(self, gamma, lmbda, eps_clip, K_epoch, entropy_coeff):
        self.make_batch()

        for _ in range(K_epoch):
            td_target = self.r + gamma * self.v(self.s_prime) * self.done

            delta = td_target - self.v(self.s)

            # Come up with a better way of doing the below calculation
            advantage_lst = []
            advantage = 0.0

            for delta_t in delta.flip(dims=[0]):
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])

            advantage_lst.reverse()

            advantage = torch.tensor(
                advantage_lst, dtype=torch.float).to(device)
            
            pi = self.pi(self.s, softmax_dim=1)

            pi_a = pi.gather(1, self.a)

            ratio = torch.exp(torch.log(pi_a) - torch.log(self.prob_a))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage

            loss = -torch.min(surr1, surr2) + \
                F.smooth_l1_loss(self.v(self.s), td_target.detach())

            # Entropy:
            # --------
            entropy = -torch.sum(pi * torch.log(pi), dim=1, keepdim=True)

            loss -= entropy_coeff * entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()


def parse_args():
    parser = argparse.ArgumentParser(
        description="PPO Training and Heatmap Generation")
    parser.add_argument('--exp_num', type=str, default="1a",
                        help='Experiment number to save results and heatmaps')
    parser.add_argument('--generate_heatmaps', action='store_true',
                        help='Generate heatmaps without training')
    parser.add_argument('--no_train', action='store_true',
                        help='Do not run training')
    parser.add_argument('--render_env', action='store_true',
                        help='renders env for every 100 episodes')
    args = parser.parse_args()

    return args

def save_hyperparams(path, hyperparams):
    with open(os.path.join(path, "hyperparameters.json"), "w") as f:
        json.dump(hyperparams, f, indent=4)


def main(args, 
         action_step_size=0.5, 
         entropy_coeff=entropy):
    # NOTE: Imported inside function because ppo and utils_ppo are circular imports
    # TODO: Change this later
    from utils_ppo import run_criticality_evolution

    if args.no_train and not args.generate_heatmaps:
        print("No operation specified. Exiting.")
        return

    exp_num = args.exp_num

    # Paths are now based on exp_num
    results_path = "ppo_training_results"
    heatmaps_result_path = "heatmaps"

    os.makedirs(results_path, exist_ok=True)

    saving_path = os.path.join(results_path, f"PPO_single_exp_{exp_num}")
    heatmaps_saving_path = os.path.join(saving_path, heatmaps_result_path)

    os.makedirs(saving_path, exist_ok=True)
    os.makedirs(heatmaps_saving_path, exist_ok=True)

    hyperparams = {
    "learning_rate": learning_rate,
    "gamma": gamma,
    "lmbda": lmbda,
    "eps_clip": eps_clip,
    "K_epoch": K_epoch,
    "num_steps": num_steps,
    "num_episodes": num_episodes,
    "save_interval": save_interval,
    "entropy_coeff": entropy,
    "action_step_size": action_step_size,
    "state_dim": state_dim,
    "action_dim": action_dim,
    "render_interval": render_interval,
    "gap_size": gap_size
    # Add any other hyperparameters here
    }

    save_hyperparams(path=saving_path, hyperparams=hyperparams)

    if not args.no_train:
        # Initialize the environment
        env = ContinuousFlatLand()

        # TODO: Reduce the width later
        env.add_obstacle([5-action_step_size, 0, 5+action_step_size, 5-gap_size])
        env.add_obstacle([5-action_step_size, 5+gap_size, 5+action_step_size, 10])

        # Discrete to continuous
        disc_to_cont_action = {0: torch.tensor((0, action_step_size), device=device, dtype=torch.float),
                               1: torch.tensor((0, -action_step_size), device=device, dtype=torch.float),
                               2: torch.tensor((action_step_size, 0), device=device, dtype=torch.float),
                               3: torch.tensor((-action_step_size, 0), device=device, dtype=torch.float)}

        # Initialize the PPO model
        model = PPO(state_dim, action_dim, learning_rate).to(device)

        overall_rewards = []
        episode_durations = []
        episode_termination_condition = []

        for n_epi in tqdm(range(num_episodes), desc="Training Progress"):
            s = env.random_reset()

            # NOTE: Normalize the state
            # s = s/10.0

            episode_reward = 0
            steps = 0

            for _ in range(num_steps):
                s_tensor = s.float().unsqueeze(0).to(device)
                prob = model.pi(s_tensor)

                # Sampling action from the policy distribution (assuming prob as distribution here; adjust as needed)
                m = Categorical(prob)
                a = m.sample().item()

                # Execute the sampled action in the environment
                # print("Before stepping environment")

                s_prime, r, done, info = env.step(disc_to_cont_action[a])


                # NOTE: Normalize the states
                # s_prime = s_prime/10.0

                # print("After stepping environment")

                if args.render_env and not n_epi % render_interval:
                    env.render()

                # model.put_data((s, a, r, s_prime, prob[0][a].item(), done))
                # Adjust as necessary
                model.put_data((s,
                                (a, 0),
                                (r, 0),
                                s_prime,
                                (prob[0][a].item(), 0),
                                (done, 0)))

                s = s_prime
                episode_reward += r
                steps += 1

                if done:
                    break

            model.train_net(gamma=gamma, 
                            lmbda=lmbda,
                            eps_clip=eps_clip, 
                            K_epoch=K_epoch, 
                            entropy_coeff=entropy_coeff)

            overall_rewards.append(episode_reward)
            episode_durations.append(steps)
            episode_termination_condition.append(
                1 if info["goal_reached"] else 0)

            if info["goal_reached"]:
                print(
                    f"State: {s_prime}, Action: {a}, Reward: {r}, Done: {done}, Info: {info}")

            if n_epi % save_interval == 0 and n_epi != 0:
                torch.save(model.state_dict(), os.path.join(
                    saving_path, f"PPO_Model_{n_epi}.pth"))
                print(
                    f"# Episode: {n_epi}, Avg Reward: {np.mean(overall_rewards[-save_interval:])}")

        env.close()
        plot_curves(saving_path, overall_rewards, episode_durations)

    if args.generate_heatmaps:
        run_criticality_evolution(
            model_directory=saving_path, heatmaps_path=heatmaps_saving_path, device=device)


def plot_curves(saving_path, overall_rewards, episode_durations):
    fig, axis = plt.subplots(2, 1, figsize=(15, 30))

    np.save(os.path.join(saving_path, "overall_rewards.npy"), np.array(overall_rewards))
    axis[0].plot(overall_rewards, color="green", label="Reward")
    axis[0].set_title("Total accumulated reward per episode")
    axis[0].set_xlabel("Episode number")
    axis[0].set_ylabel("Reward")

    np.save(os.path.join(saving_path, "total_steps.npy"), np.array(episode_durations))
    axis[1].plot(episode_durations, color="blue", label="Steps per episode")
    axis[1].set_title("Total steps per episode")
    axis[1].set_xlabel("Episode number")
    axis[1].set_ylabel("Steps")

    # axis[2].plot(entropy_coeff_decay, color="red", label="Entropy coeff decay")
    # axis[2].set_title("Entropy coefficient decay")
    # axis[2].set_xlabel("Episode number")
    # axis[2].set_ylabel("Entropy coefficient")

    fig.savefig(os.path.join(saving_path, "training_plots.png"))
    plt.show()


if __name__ == '__main__':
    """
    To run the script

    call python ppo.py

    the available args parser are

    --exp_num [desired exp number]
    -- no_train --> if you dont want training to happen
    -- generate_heatmaps --> if you want to generate heat maps

    Eg. usage

    1. In case if you want to train and generate heat maps --> call python ppo.py --exp_num 1c --generate_heatmaps
    2. In case if you just want to train --> call python ppo.py --exp_num 1c
    3. In case if you want to just generate heatmaps --> call python ppo.py --exp_num 1c --no_train --generate_heatmaps
    """
    args = parse_args()
    main(args=args)
