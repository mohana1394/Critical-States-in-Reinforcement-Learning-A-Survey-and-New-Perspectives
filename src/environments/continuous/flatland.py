# Imports:
# --------
import torch
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# Continuous FlatLand GPU compatible:
# -----------------------------------
class ContinuousFlatLand(gym.Env):
    def __init__(self, delta_action: float = 1.0, device='cpu'):
        super(ContinuousFlatLand, self).__init__()
        self.delta_action = delta_action
        self.device = torch.device(device)

        # Using NumPy arrays for spaces definition as required by Gymnasium
        self.action_space = spaces.Box(low=np.array([-self.delta_action, -self.delta_action], dtype=np.float32),
                                       high=np.array(
                                           [self.delta_action, self.delta_action], dtype=np.float32),
                                       dtype=np.float32)

        self.observation_space = spaces.Box(low=np.array([0.0, 0.0], dtype=np.float32),
                                            high=np.array(
                                                [10.0, 10.0], dtype=np.float32),
                                            dtype=np.float32)

        self.state = torch.tensor(
            [0.0, 10.0], device=self.device, dtype=torch.float32)  # Default state
        self.goal = torch.tensor(
            [9.85, 0.2], device=self.device, dtype=torch.float32)  # Goal location

        self.obstacles = []

        self.figure, self.axis = plt.subplots()
        plt.ion()

    def add_obstacle(self, obstacle):
        self.obstacles.append(torch.tensor(
            obstacle, device=self.device, dtype=torch.float32))

    def check_collision(self, new_state):
        for obstacle in self.obstacles:
            if obstacle[0] <= new_state[0] <= obstacle[2] and obstacle[1] <= new_state[1] <= obstacle[3]:
                return True
        return False

    def step(self, action):
        # action = torch.tensor(action, device=self.device, dtype=torch.float32)
        # action = torch.clamp(action, -self.delta_action, self.delta_action)
        new_state = self.state + action.to(self.device)

        collision = self.check_collision(new_state)

        distance_to_goal = torch.norm(new_state - self.goal)
        goal_reached = distance_to_goal < self.delta_action

        if collision:
            reward, done = -1, True
        else:
            self.state = torch.clamp(new_state, 0.0, 10.0)

            # Modified reward structure
            reward, done = (+10, True) if goal_reached else (0, False)

        info = {
            'is_collision': collision,
            'goal_reached': goal_reached,
            'distance_to_goal': distance_to_goal.item(),
        }

        return self.state, reward, done, info
    
    # NOTE: We use fixed reset for DQN
    def fixed_reset(self,
                    reset_state=(0.0, 10.0)):
        
        self.state = torch.tensor(reset_state, 
                                  device=self.device, 
                                  dtype=torch.float32)
        
        return self.state.unsqueeze(0)

    # NOTE: We use random reset for PPO and SAC  
    def random_reset(self):
        reset_state = random.choice([(
            random.choice([round(random.uniform(0, 4), 1),
                          round(random.uniform(6, 9), 1)]),
            random.choice([round(random.uniform(0, 4), 1),
                          round(random.uniform(6, 9), 1)])
        ),
            (round(random.uniform(5-0.5, 5+0.5), 1),
             round(random.uniform(5-0.1, 5+0.1), 1))
        ])

        self.state = torch.tensor(reset_state, 
                                  device=self.device, 
                                  dtype=torch.float32)

        return self.state.unsqueeze(0)

    def render(self, refresh_freq=0.01):
        self.axis.clear()
        self.axis.set_xlim(0.0, 10.0)
        self.axis.set_ylim(0.0, 10.0)

        for obstacle in self.obstacles:
            self.axis.add_patch(patches.Rectangle(
                (obstacle[0].item(), obstacle[1].item()),
                (obstacle[2] - obstacle[0]).item(),
                (obstacle[3] - obstacle[1]).item(),
                linewidth=1, edgecolor='k', facecolor='red'))

        self.axis.plot(self.goal[0].item(),
                       self.goal[1].item(), 'gs', markersize=9)
        self.axis.plot(self.state[0].item(),
                       self.state[1].item(), 'bs', markersize=5)

        plt.pause(refresh_freq)

    def close(self):
        plt.close()


# Test environment:
# -----------------
if __name__ == "__main__":
    # Usage
    env = ContinuousFlatLand(device='cuda')  # Change to 'cuda' for GPU
    # env.add_obstacle([4.5, 0, 5.5, 3.5])  # Define obstacle
    # env.add_obstacle([4.5, 6.5, 5.5, 10])

    env.add_subgoal([4.5, 3.5, 5.5, 6.5])

    state = env.reset()
    for _ in range(100_000):
        action = torch.tensor(env.action_space.sample(),
                              device=env.device)  # Random action
        state, reward, done, info = env.step(action)

        print(
            f"State: {state}, Action: {action}, Reward: {reward}, Done: {done}, Info: {info}")

        # env.render(refresh_freq=0.01)
        if done:
            break

    plt.savefig("Continuous_Flat_Land.png")

    env.close()
