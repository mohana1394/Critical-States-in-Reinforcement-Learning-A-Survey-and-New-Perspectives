# Imports:
# --------
import os
import math
import torch
import random
import numpy as np
import scienceplots
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.nn import functional as F
from src.algorithms.sac.sac_model import PolicyNet, QNet
from src.environments.continuous.flatland import ContinuousFlatLand


# Class 1: Replay Buffer Class
# --------
class replayBuffer:
    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        self.buffer = []
        self._next_idx = 0

    def add(self, item):
        if len(self.buffer) > self._next_idx:
            self.buffer[self._next_idx] = item
        else:
            self.buffer.append(item)
        if self._next_idx == self.buffer_size - 1:
            self._next_idx = 0
        else:
            self._next_idx = self._next_idx + 1

    def sample(self, batch_size):
        indices = [random.randint(0, len(self.buffer) - 1) for _ in range(batch_size)]
        states   = [self.buffer[i][0] for i in indices]
        actions  = [self.buffer[i][1] for i in indices]
        rewards  = [self.buffer[i][2] for i in indices]
        n_states = [self.buffer[i][3] for i in indices]
        dones    = [self.buffer[i][4] for i in indices]
        return states, actions, rewards, n_states, dones

    def length(self):
        return len(self.buffer)


# Function 1: Choose action
# -----------
def pick_sample(s,
                pi_model,
                device):

    with torch.no_grad():
        #   --> size : (1, 4)
        # s_batch = np.expand_dims(s, axis=0)
        # s_batch = torch.tensor(s_batch, dtype=torch.float).to(device)
        # Get logits from state
        #   --> size : (1, 2)
        logits = pi_model(s.expand(1, 2).to(device))
        #   --> size : (2)
        logits = logits.squeeze(dim=0)
        # From logits to probabilities
        probs = F.softmax(logits, dim=-1)
        # Pick up action's sample
        #   --> size : (1)
        a = torch.multinomial(probs, num_samples=1)
        #   --> size : ()
        a = a.squeeze(dim=0)

        # Return
        return a.tolist()


# Class 2:
# --------
class categorical:
    def __init__(self, s, pi_model):
        logits = pi_model(s)
        self._prob = F.softmax(logits, dim=-1)
        self._logp = torch.log(self._prob)

    # probability (sum is 1.0) : P
    def prob(self):
        return self._prob

    # log probability : log P()
    def logp(self):
        return self._logp


# Function 2: Optimize
# -----------
def optimize_theta(states,
                   alpha,
                   opt_pi,
                   pi_model,
                   q_origin_model1,
                   device):

    # Convert to tensor
    states = torch.tensor(states, dtype=torch.float).to(device)
    # Disable grad in q_origin_model1 before computation
    # (or use q_value.detach() not to include in graph)
    for p in q_origin_model1.parameters():
        p.requires_grad = False

    # Optimize
    opt_pi.zero_grad()
    dist = categorical(states, pi_model)
    q_value = q_origin_model1(states)
    term1 = dist.prob()
    term2 = q_value - alpha * dist.logp()
    expectation = term1.unsqueeze(dim=1) @ term2.unsqueeze(dim=2)
    expectation = expectation.squeeze(dim=1)
    (-expectation).sum().backward()
    opt_pi.step()
    # Enable grad again
    for p in q_origin_model1.parameters():
        p.requires_grad = True


# Function 3:
# -----------
def optimize_phi(alpha,
                 gamma,
                 states,
                 actions,
                 rewards,
                 next_states,
                 dones,
                 opt_q1,
                 opt_q2,
                 pi_model,
                 q_origin_model1,
                 q_origin_model2,
                 q_target_model1,
                 q_target_model2,
                 device):
    #
    # Convert to tensor
    #
    states = torch.tensor(states, dtype=torch.float).to(device)
    actions = torch.tensor(actions, dtype=torch.int64).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float).to(device)
    rewards = rewards.unsqueeze(dim=1)
    next_states = torch.tensor(next_states, dtype=torch.float).to(device)
    dones = torch.tensor(dones, dtype=torch.float).to(device)
    dones = dones.unsqueeze(dim=1)

    #
    # Compute r + gamma * (1 - d) (min Q(s_next,a_next') + alpha * H(P))
    #
    with torch.no_grad():
        # min Q(s_next,a_next')
        q1_tgt_next = q_target_model1(next_states)
        q2_tgt_next = q_target_model2(next_states)
        dist_next = categorical(next_states, pi_model)
        q1_target = q1_tgt_next.unsqueeze(
            dim=1) @ dist_next.prob().unsqueeze(dim=2)
        q1_target = q1_target.squeeze(dim=1)
        q2_target = q2_tgt_next.unsqueeze(
            dim=1) @ dist_next.prob().unsqueeze(dim=2)
        q2_target = q2_target.squeeze(dim=1)
        q_target_min = torch.minimum(q1_target, q2_target)
        # alpha * H(P)
        h = dist_next.prob().unsqueeze(dim=1) @ dist_next.logp().unsqueeze(dim=2)
        h = h.squeeze(dim=1)
        h = -alpha * h
        # total
        term2 = rewards + gamma * (1.0 - dones) * (q_target_min + h)

    #
    # Optimize critic loss for Q-network1
    #
    opt_q1.zero_grad()
    # !num_classes was changed from 2 to 4
    one_hot_actions = F.one_hot(actions, num_classes=4).float()
    q_value1 = q_origin_model1(states)
    term1 = q_value1.unsqueeze(dim=1) @ one_hot_actions.unsqueeze(dim=2)
    term1 = term1.squeeze(dim=1)
    loss_q1 = F.mse_loss(
        term1,
        term2,
        reduction="none")
    loss_q1.sum().backward()
    opt_q1.step()

    #
    # Optimize critic loss for Q-network2
    #
    opt_q2.zero_grad()
    # !num_classes was changed from 2 to 4
    one_hot_actions = F.one_hot(actions, num_classes=4).float()
    q_value2 = q_origin_model2(states)
    term1 = q_value2.unsqueeze(dim=1) @ one_hot_actions.unsqueeze(dim=2)
    term1 = term1.squeeze(dim=1)
    loss_q2 = F.mse_loss(
        term1,
        term2,
        reduction="none")
    loss_q2.sum().backward()
    opt_q2.step()


# Function 4: Update target network
# -----------
def update_target(tau,
                  q_origin_model1,
                  q_origin_model2,
                  q_target_model1,
                  q_target_model2):

    for var, var_target in zip(q_origin_model1.parameters(), q_target_model1.parameters()):
        var_target.data = tau * var.data + (1.0 - tau) * var_target.data
    for var, var_target in zip(q_origin_model2.parameters(), q_target_model2.parameters()):
        var_target.data = tau * var.data + (1.0 - tau) * var_target.data


# Function 5: Plot training curves
# -----------
def plot_training_curves(reward_records,
                         save_path):
    # Set plot style to Scientific Plots
    plt.style.use(['science', 'ieee'])

    # Generate recent 50 interval average
    average_reward = []
    for idx in range(len(reward_records)):
        avg_list = np.empty(shape=(1,), dtype=int)
        if idx < 50:
            avg_list = reward_records[:idx+1]
        else:
            avg_list = reward_records[idx-49:idx+1]
        average_reward.append(np.average(avg_list))

    plt.plot(reward_records, label="Rewards")
    plt.plot(average_reward, label="Average reward")
    plt.legend()
    plt.xlabel("Rewards")
    plt.ylabel("Episodes")
    plt.savefig(os.path.join(save_path, "Training_curves.png"),
                dpi=300,
                bbox_inches='tight')
    plt.close()


# Function 6: Main train function
# -----------
def train_sac(no_episodes,
              max_buffer_size,
              max_steps,
              batch_size,
              alpha,
              learning_rate,
              gamma,
              tau,
              action_step_size=0.5,
              render=False,
              save_interval=100,
              env_gap_size=0.15,
              env_random_reset=True,
              save_path="SAC",
              device="cuda:0"):

    # Step 1: Define everything you need
    # -------
    env = ContinuousFlatLand()
    env.add_obstacle(
        [5-action_step_size, 0, 5+action_step_size, 5-env_gap_size])
    env.add_obstacle(
        [5-action_step_size, 5+env_gap_size, 5+action_step_size, 10])

    reward_records = []
    goal_reached = []
    buffer = replayBuffer(max_buffer_size)

    disc_to_cont_action = {0: torch.tensor((0, action_step_size), device=device, dtype=torch.float),
                           1: torch.tensor((0, -action_step_size), device=device, dtype=torch.float),
                           2: torch.tensor((action_step_size, 0), device=device, dtype=torch.float),
                           3: torch.tensor((-action_step_size, 0), device=device, dtype=torch.float)}

    pi_model = PolicyNet().to(device)

    q_origin_model1 = QNet().to(device)  # Q_phi1
    q_origin_model2 = QNet().to(device)  # Q_phi2
    q_target_model1 = QNet().to(device)  # Q_phi1'
    q_target_model2 = QNet().to(device)  # Q_phi2'
    _ = q_target_model1.requires_grad_(False)  # target model doen't need grad
    _ = q_target_model2.requires_grad_(False)  # target model doen't need grad

    opt_pi = torch.optim.AdamW(pi_model.parameters(), lr=learning_rate)
    opt_q1 = torch.optim.AdamW(q_origin_model1.parameters(), lr=learning_rate)
    opt_q2 = torch.optim.AdamW(q_origin_model2.parameters(), lr=learning_rate)

    # Step 2: Create required directories
    # -------
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
        os.mkdir(os.path.join(save_path, "models"))


    # Train loop:
    # -----------
    for epi_no in tqdm(range(no_episodes), desc="Episode", colour="blue"):
        
        if env_random_reset:
            s = env.random_reset()
        else:
            s = env.fixed_reset()

        done = False
        cum_reward = 0

        for _ in range(max_steps):
            a = pick_sample(s=s,
                            pi_model=pi_model,
                            device=device)
            """
            S is a numpy array and has shape (4,)
            A is just one integer (0,1,2,3) - You can directly use it
            """
            s_next, r, done, info = env.step(disc_to_cont_action[a])

            """
            S_next: ndarray (4,)
            Reward: int
            term: boolean
            """
            #! Render:
            if render:
                env.render()

            buffer.add([s.tolist(), a, r, s_next.tolist(), float(done)])
            cum_reward += r

            if buffer.length() >= batch_size:
                states, actions, rewards, n_states, dones = buffer.sample(
                    batch_size)

                optimize_theta(states=states,
                               alpha=alpha,
                               opt_pi=opt_pi,
                               pi_model=pi_model,
                               q_origin_model1=q_origin_model1,
                               device=device)

                optimize_phi(alpha=alpha,
                             gamma=gamma,
                             states=states,
                             actions=actions,
                             rewards=rewards,
                             next_states=n_states,
                             dones=dones,
                             opt_q1=opt_q1,
                             opt_q2=opt_q2,
                             pi_model=pi_model,
                             q_origin_model1=q_origin_model1,
                             q_origin_model2=q_origin_model2,
                             q_target_model1=q_target_model1,
                             q_target_model2=q_target_model2,
                             device=device)

                
                update_target(tau=tau,
                              q_origin_model1=q_origin_model1,
                              q_origin_model2=q_origin_model2,
                              q_target_model1=q_target_model1,
                              q_target_model2=q_target_model2)

            s = s_next

            if done:
                break

        output_info = f"Episode no = {epi_no}, Cumulative reward = {cum_reward}"
        goal_reached.append(info["goal_reached"])

        if info["goal_reached"]:
            tqdm.write(f"{output_info}, Reached goal!")
        elif info["is_collision"]:
            print(f"{output_info}, Reached danger zone!")
        else:
            print(f"{output_info}, Ran out of time!")

        # Save model:
        # -----------
        if (epi_no % save_interval == 0) and (epi_no != 0):
            # Save policy network:
            torch.save(pi_model.state_dict(),
                       os.path.join(save_path, "models",  f"PolicyNet_{epi_no}.pth"))
            # Save Q-Net original 1:
            torch.save(q_origin_model1.state_dict(),
                       os.path.join(save_path, "models",  f"QNet1_{epi_no}.pth"))
            # Save Q-Net original 2:
            torch.save(q_origin_model2.state_dict(),
                       os.path.join(save_path, "models", f"QNet2_{epi_no}.pth"))

        reward_records.append(cum_reward)


    env.close()

    # Plot and save training curves:
    np.save(os.path.join(save_path, "overall_rewards.npy"), np.array(reward_records))
    np.save(os.path.join(save_path, "episode termination_condition.npy"), np.array(goal_reached))

    plot_training_curves(reward_records=reward_records,
                         save_path=save_path)


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


# Function 8: Calculate Q variance
# -----------
def cal_q_variance(q_values):
    """
    Calculates variance of the given Q-values

    return: numpy array of Q variance (10_000, n_metrics)
    """
    return np.var(q_values, axis=1)


# Function 9: Calculate Delta Q
# -----------
def cal_delta_q(q_values):
    """
    Calculates delta Q (maxQ - minQ) of the given Q-values

    return: numpy array of delta Q (10_000, n_metrics)
    """
    return np.max(q_values, axis=1) - np.min(q_values, axis=1)


# Function 10: Criticality metrics list
# ------------
def cal_criticality():
    """
    This function has a list of all criticality metrics
    # TODO: Add more criticality metrics
    """
    return {"q_variance": cal_q_variance,
            "delta_q": cal_delta_q}


# Function 11: Generates heatmaps per dataset per model
# ------------
def generate_heatmap(data_points,
                     q_values,
                     index,
                     type,
                     saving_path):
    """
    Creates heatmaps for different criticality metrics.

    output: saves heatmaps as .PNG files
    """
    # Set style to scientific plots:
    plt.style.use(['science', 'ieee'])

    criticality_metrics = cal_criticality()

    for name, each_metric in criticality_metrics.items():
        criticality = each_metric(q_values)
        plt.scatter(data_points[:, 0],
                    data_points[:, 1],
                    c=criticality,
                    alpha=0.3,
                    cmap="hot_r")
        plt.colorbar()
        
        plt.savefig(os.path.join(saving_path, f"{type}_{name}_{index}.png"), 
                    dpi=300, 
                    bbox_inches='tight')
        plt.close()


# Function 12: Evolution of heatmaps
# ------------
def heatmap_evolution_sac(device,
                      model_path,
                      saving_path,
                      batch_size: int = 250,
                      freq: int = 100,
                      n_observations: int = 2,
                      n_actions: int = 4):
    """
    Function takes snapshots of saved models and creates a GIF image of the evolution

    output: saves evolution of heatmaps as a .GIF image
    """
    data_points = generate_test_points()

    # Step 1: Get model indices
    # -------
    files = [each_file for each_file in os.listdir(
        model_path) if each_file.endswith('.pth')]

    policynet_files = [
        each_file for each_file in files if "PolicyNet" in each_file]
    qnet1_files = [each_file for each_file in files if "QNet1" in each_file]
    qnet2_files = [each_file for each_file in files if "QNet2" in each_file]

    policy_index = {int(each_file.split("_", -1)[-1].split(".")[0]): each_file
                    for each_file in policynet_files if not int(each_file.split("_", -1)[-1].split(".")[0]) % freq}
    qnet1_index = {int(each_file.split("_", -1)[-1].split(".")[0]): each_file
                   for each_file in qnet1_files if not int(each_file.split("_", -1)[-1].split(".")[0]) % freq}
    qnet2_index = {int(each_file.split("_", -1)[-1].split(".")[0]): each_file
                   for each_file in qnet2_files if not int(each_file.split("_", -1)[-1].split(".")[0]) % freq}

    # Prompt user to pass correct frequency values:
    if (len(policy_index) == 1) or (len(qnet1_index) == 1) or (len(qnet2_index) == 1):
        print("Check the save frequency.")

    # Step 2: Create a directory for saving
    # ------
    saving_path = os.path.join(saving_path, "heatmaps")

    if not os.path.isdir(saving_path):
        os.mkdir(saving_path)

    # Step 2: Loop through the models
    # -------
    for index in policy_index:
        policynet = PolicyNet()
        qnet1 = QNet()
        qnet2 = QNet()

        policynet.load_state_dict(torch.load(os.path.join(model_path, policy_index[index]),
                                       map_location=torch.device(device)))
        qnet1.load_state_dict(torch.load(os.path.join(model_path, qnet1_index[index]),
                                       map_location=torch.device(device)))
        qnet2.load_state_dict(torch.load(os.path.join(model_path, qnet2_index[index]),
                                       map_location=torch.device(device)))

        policynet.eval()
        qnet1.eval()
        qnet2.eval()

        with torch.no_grad():
            policy_values, q1_values, q2_values = [], [], []

            for mini_batch in range(0, data_points.shape[0], batch_size):
                mini_policy = policynet(data_points[mini_batch:mini_batch+batch_size])
                mini_policy = F.softmax(mini_policy, dim=-1)

                mini_q1_values = qnet1(data_points[mini_batch:mini_batch+batch_size])
                mini_q2_values = qnet2(data_points[mini_batch:mini_batch+batch_size])
                
                policy_values.append(mini_policy.numpy())
                q1_values.append(mini_q1_values.numpy())
                q2_values.append(mini_q2_values.numpy())

            policy_values = np.concatenate(policy_values)
            q1_values = np.concatenate(q1_values, axis=0)
            q2_values = np.concatenate(q2_values, axis=0)

        # Policy heatmaps:
        generate_heatmap(data_points=data_points,
                         q_values=policy_values,
                         index=index,
                         type="policy",
                         saving_path=saving_path)
        
        generate_heatmap(data_points=data_points,
                         q_values=q1_values,
                         index=index,
                         type="q1",
                         saving_path=saving_path)
    
        generate_heatmap(data_points=data_points,
                         q_values=q2_values,
                         index=index,
                         type="q2",
                         saving_path=saving_path)
