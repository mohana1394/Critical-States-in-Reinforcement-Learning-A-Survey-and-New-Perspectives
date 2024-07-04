# Imports:
# --------
import os
import math
import torch
import random
import numpy as np
import scienceplots
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib import animation
from collections import deque, namedtuple


# Module imports:
# ---------------
from src.algorithms.dqn.dqn_model import DQN
from src.environments.continuous.flatland import ContinuousFlatLand


# Class 1: Replay Buffer
# -------
class ReplayMemory(object):
    """
    Class to create a Replay Memory to train DQN
    """

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, item):
        """Save a transition"""
        self.memory.append(item)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# Function 2: Select actions
# -----------
def select_action(policy_net,
                  state,
                  episode_eps,
                  device="cuda:0"):
    """
    Function decides to Exploit or Explore and chooses an action.
    """
    sample = random.random()

    if sample > episode_eps:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)

    else:
        return torch.tensor([[np.random.choice([0, 1, 2, 3])]], device=device, dtype=torch.long)


# Function 3: Create GIFs of the environment
# -----------
def animate(env,
            configs,
            interval=50):
    """helper function to create demo animations"""

    fig, ax = plt.subplots()

    _, ax, _, _ = env.render(ax=ax)

    count = 1

    def func(i):
        nonlocal count
        env.set_config(*configs[i])
        ax.clear()
        env.render(ax=ax)
        fig.canvas.draw()

        ax.set_title(f"Step number: {count}")

        count += 1

    ax.set_xticks([])
    ax.set_yticks([])

    plt.close()

    anim = animation.FuncAnimation(fig, func, len(configs), interval=interval)

    return anim


# Function 4: One step of optimizing the model
# -----------
def optimize_model(policy_net,
                   target_net,
                   optimizer,
                   memory,
                   batch_size,
                   gamma,
                   train_per_replay_mem,
                   device):
    """
    Function does one step of Policy Network updates when called.

    Modified from Pytorch's official implemetation of DQN for CartPole-v1
    URL: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    """

    Transition = namedtuple('Transition',
                            ('state', 'action', 'next_state', 'reward', 'done'))

    if len(memory) < (batch_size*train_per_replay_mem):
        return

    # NOTE: We are sampling multiple times from the same replay buffer before updating
    for _ in range(train_per_replay_mem):
        transitions = memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        done_batch = torch.cat(batch.done)

        state_action_values = policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(batch_size, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(
                non_final_next_states).max(1).values

        # NOTE: We are multiplying expected_state_action_values with 'done' flag
        """
        If done=True DON'T consider the next reward
        """
        expected_state_action_values = (
            next_state_values * gamma * done_batch) + reward_batch

        # Compute Huber loss:
        # -------------------
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values,
                         expected_state_action_values.unsqueeze(1))

        # Optimize the model:
        # -------------------
        optimizer.zero_grad()
        loss.backward()

        # In-place gradient clipping:
        # ---------------------------
        # NOTE: Clipping parameter 1.0 gave the best results
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 1.0)
        optimizer.step()


# Function 5:
# -----------
def plot_curves(saving_path,
                overall_rewards,
                episode_durations,
                episode_termination_condition,
                epsilon_decay)->None:
    """
    Function plots the training curves after training is finished and saves plots in the saving_path
    """
    plt.style.use(['science', 'ieee'])

    fig1, axis1 = plt.subplots(1, 1, figsize=(12, 8))
    axis1.plot([episode for episode in range(1, len(overall_rewards)+1)],
                    overall_rewards, color="blue", label="Reward")
    axis1.set_title("Total accumulated reward per episode")
    axis1.set_xlabel("Episode number")
    axis1.set_ylabel("Reward")
    fig1.savefig(os.path.join(saving_path, "Rewards.png"))
    plt.close()


    fig2, axis2 = plt.subplots(1, 1, figsize=(12, 8))
    axis2.plot([episode for episode in range(1, len(episode_durations)+1)],
                    episode_durations, color="blue", label="Steps per episode")
    axis2.set_title("Total steps per episode")
    axis2.set_xlabel("Episode number")
    axis2.set_ylabel("Steps")
    fig2.savefig(os.path.join(saving_path, "Steps.png"))
    plt.close()

    fig3, axis3 = plt.subplots(1, 1, figsize=(12, 8))
    axis3.plot([episode for episode in range(1, len(episode_termination_condition)+1)],
                    episode_termination_condition, color="blue", label="Goal reached!")
    axis3.set_title("Goal reached!")
    axis3.set_xlabel("Episode number")
    axis3.set_ylabel("Successful or not \n(Yes=1, No=0)")
    fig3.savefig(os.path.join(saving_path, "Termination_condition.png"))
    plt.close()

    fig4, axis4 = plt.subplots(1, 1, figsize=(12, 8))
    axis4.plot([eps for eps in range(1, len(epsilon_decay)+1)],
                    epsilon_decay, color="blue", label="Episode epsilon")
    axis4.set_title("Episode epsilon")
    axis4.set_xlabel("Episode number")
    axis4.set_ylabel("Epsilon")
    fig4.savefig(os.path.join(saving_path, "Epsilon_decay.png"))
    plt.close()


# Function 6: Main DQN training function
# ----------
def train_dqn(device,
              render,
              saving_path,
              train_per_replay_mem,
              saving_freq,
              batch_size,
              num_episodes,
              max_episode_duration,
              gamma,
              max_replay_mem,
              n_observations,
              n_actions,
              learning_rate,
              target_update_rate,
              eps_start,
              eps_end,
              eps_decay,
              env_random_reset,
              action_step_size=0.5,
              env_gap_size=0.15):
    """
    Function is the main function to call to train the DQN in continuous grid environment 
    """

    # Step 1: Create a folder to save results
    # -------
    if not os.path.isdir(saving_path):
        os.mkdir(os.path.join(saving_path))
        os.mkdir(os.path.join(saving_path, "models"))

    # Step 2: Select device
    # -------
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    print(f"Device name: {torch.cuda.get_device_name()}")

    # Step 3: Create necessary instances
    # -------
    episode_durations = []
    overall_rewards = []
    episode_termination_condition = []
    epsilon_decay = []

    Transition = namedtuple('Transition',
                            ('state', 'action', 'next_state', 'reward', 'done'))
    memory = ReplayMemory(max_replay_mem)

    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    # NOTE: Changed the optimizer to Adam from AdamW
    optimizer = optim.Adam(policy_net.parameters(),
                           lr=learning_rate,
                           amsgrad=True)

    # Step 4: Create the continuous flatland environment
    # -------
    env = ContinuousFlatLand(device=device)
    # Define obstacles
    env.add_obstacle([5-action_step_size, 0, 5+action_step_size, 5-env_gap_size])
    env.add_obstacle([5-action_step_size, 5+env_gap_size, 5+action_step_size, 10])

    # Step 5: Define discrete to continuous action mapping
    # ------
    disc_to_cont_action = {0: torch.tensor((0, action_step_size), device=device, dtype=torch.float),
                           1: torch.tensor((0, -action_step_size), device=device, dtype=torch.float),
                           2: torch.tensor((action_step_size, 0), device=device, dtype=torch.float),
                           3: torch.tensor((-action_step_size, 0), device=device, dtype=torch.float)}

    # Step 6: Define a reward mapping
    # -------
    # NOTE: In case of DQN, +1 and -1 rewards worked
    reward_mapping = {-1: torch.tensor([-1], device=device),
                      +10: torch.tensor([+10], device=device),
                      0: torch.tensor([0], device=device)}

    # Step 7: Done mapping
    # -------
    done_mapping = {True: torch.tensor([0], device=device),
                    False: torch.tensor([1], device=device)}

    # Step 8: Training loop
    # -------
    for i_episode in tqdm(range(num_episodes), desc="Episode no", colour="green"):
        episode_eps = eps_end + (eps_start - eps_end) * \
            math.exp(-1. * i_episode / eps_decay)

        episode_rewards = []

        # Choose between random reset and fixed reset:
        if not env_random_reset:
            state = env.fixed_reset()
            state = state.unsqueeze(0)
        else:
            state = env.random_reset()
            state = state.unsqueeze(0)

        # Normalize states
        # ----------------
        # NOTE: Normalization gave the best results
        state = state/10.0

        for step_num in range(max_episode_duration):
            action = select_action(policy_net=policy_net,
                                   state=state,
                                   episode_eps=episode_eps,
                                   device=device)

            # NOTE: While taking action we need a numpy array of (x, y)
            observation, reward, terminated, info = env.step(
                disc_to_cont_action[action.item()])

            # NOTE: Choose if you want to render or not
            if render:
                env.render()

            # Normalize observation
            observation = tuple(each_item/10.0 for each_item in observation)

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation,
                                          dtype=torch.float32,
                                          device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(Transition(state,
                                   action,
                                   next_state,
                                   reward_mapping[reward],
                                   done_mapping[terminated]))

            # Move to the next state
            state = next_state

            optimize_model(policy_net=policy_net,
                           target_net=target_net,
                           optimizer=optimizer,
                           memory=memory,
                           batch_size=batch_size,
                           gamma=gamma,
                           train_per_replay_mem=train_per_replay_mem,
                           device=device)

            # Soft update of the target network's weights:
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()

            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * target_update_rate + \
                    target_net_state_dict[key]*(1-target_update_rate)

            target_net.load_state_dict(target_net_state_dict)

            episode_rewards.append(reward)

            # Decision to terminate the episode
            if terminated or (step_num+1 == max_episode_duration):
                episode_durations.append(step_num+1)
                break

        # Gather the last eps_threshold
        epsilon_decay.append(episode_eps)

        # Gather information of the episode for plotting
        overall_rewards.append(np.sum(episode_rewards))

        episode_termination_condition.append(
            1 if info["goal_reached"] else 0)

        # Logging for user:
        tqdm.write(f"Episode {i_episode+1}: {info} ")

        # Save model based on a saving frequency
        if not i_episode % saving_freq:
            torch.save(policy_net.state_dict(),
                       os.path.join(saving_path, "models", f"DQN_chk_point_{i_episode}.pt"))

    env.close()

    # Save the training curves:
    np.save(os.path.join(saving_path, "overall_rewards.npy"), np.array(overall_rewards))
    np.save(os.path.join(saving_path, "episode_duration.npy"), np.array(episode_durations))
    np.save(os.path.join(saving_path, "episode_termination_condition.npy"), np.array(episode_termination_condition))
    np.save(os.path.join(saving_path, "epsilon_decay.npy"), np.array(epsilon_decay))

    # Plot the training curves
    plot_curves(saving_path=saving_path,
                overall_rewards=overall_rewards,
                episode_durations=episode_durations,
                episode_termination_condition=episode_termination_condition,
                epsilon_decay=epsilon_decay)


# Heatmaps:
# ---------

# Function 7: Generate test data points for heatmaps
# -----------
def generate_test_points(min: float = 0.0,
                         max: float = 10.0,
                         num_points: int = 8_000):

    num_points_sqrt = math.sqrt(num_points)

    if not num_points_sqrt.is_integer():
        print(
            f"Number of points rounded to the nearest exact square: {int(num_points_sqrt)}")

    x_points, y_points = np.linspace(min, max, int(num_points_sqrt)), \
        np.linspace(min, max, int(num_points_sqrt))

    data_points = np.array([(x, y) for x in x_points for y in y_points])

    return torch.from_numpy(data_points).to(torch.float32)


# Function 9: Calculate Q variance
# -----------
def cal_q_variance(q_values):
    """
    Calculates variance of the given Q-values

    return: numpy array of Q variance (10_000, n_metrics)
    """
    return np.var(q_values, axis=1)


# Function 10: Calculate Delta Q
# -----------
def cal_delta_q(q_values):
    """
    Calculates delta Q (maxQ - minQ) of the given Q-values

    return: numpy array of delta Q (10_000, n_metrics)
    """
    return np.max(q_values, axis=1) - np.min(q_values, axis=1)


# Function 11:
# ------------
def cal_skewness(q_values):
    """
    Calculates Pearson's second skewness coefficient of the given Q-values

    return: numpy array of skewness of Q (10_000, n_metrics)
    """
    return 3*(np.mean(q_values, axis=1)-np.median(q_values, axis=1))/(np.std(q_values, axis=1))


def cal_spread(q_values):
    """
    Calculates spread of the given Q-values as given by IROS paper

    return: numpy array of spread of Q (10_000, n_metrics)
    """
    return np.max(q_values, axis=1) - np.mean(q_values, axis=1)


# Function 11: Criticality metrics list
# ------------
def cal_criticality():
    """
    This function has a list of all criticality metrics
    """
    return {"q_variance": cal_q_variance,
            "delta_q": cal_delta_q,
            "skewness_q": cal_skewness,
            "spread_q": cal_spread}


# Function 12: Generates heatmaps per dataset per model
# ------------
def generate_heatmap(data_points,
                     q_values,
                     index,
                     saving_path):
    """
    Creates heatmaps for different criticality metrics.

    output: saves heatmaps as .PNG files
    """

    criticality_metrics = cal_criticality()

    for name, each_metric in criticality_metrics.items():
        criticality = each_metric(q_values)

        plt.style.use(['science', 'ieee'])
        plt.scatter(data_points[:, 0],
                    data_points[:, 1],
                    c=criticality,
                    alpha=0.3,
                    cmap="hot_r")
        plt.colorbar()
        plt.savefig(os.path.join(
            saving_path, f"{name}_{index}.png"), dpi=300, bbox_inches='tight')
        plt.close()


# Function 11: Evolution of heatmaps
# ------------
def heatmap_evolution_dqn(device,
                      model_path,
                      saving_path,
                      batch_size: int = 128,
                      freq: int = 1_000,
                      n_observations: int = 2,
                      n_actions: int = 4):
    """
    Function takes snapshots of saved models and creates a GIF image of the evolution

    output: saves evolution of heatmaps as a .GIF image
    """
    data_points = generate_test_points()

    # Step 1: Get model indices
    # -------
    files = os.listdir(model_path)

    model_index = {int(each_file.split("_", -1)[-1].split(".")[0]): each_file
                   for each_file in files if not int(each_file.split("_", -1)[-1].split(".")[0]) % freq}

    # Prompt user to pass correct frequency values:
    if len(model_index) == 1:
        print("Check the save frequency.")

    # Step 2: Create a directory for saving
    # ------
    if not os.path.isdir(saving_path):
        os.mkdir(os.path.join(saving_path))

    # Step 2: Loop through the models
    # -------
    for index in model_index:
        dqn = DQN(n_observations=n_observations,
                  n_actions=n_actions)
        dqn.load_state_dict(torch.load(os.path.join(model_path, model_index[index]),
                                       map_location=torch.device(device)))

        dqn.eval()

        with torch.no_grad():
            q_values = []
            for mini_batch in range(0, data_points.shape[0], batch_size):
                mini_qvalue = dqn(
                    data_points[mini_batch:mini_batch+batch_size]/10.0)
                q_values.append(mini_qvalue.numpy())

            q_values = np.concatenate(q_values, axis=0)

        generate_heatmap(data_points=data_points,
                         q_values=q_values,
                         index=index,
                         saving_path=saving_path)
