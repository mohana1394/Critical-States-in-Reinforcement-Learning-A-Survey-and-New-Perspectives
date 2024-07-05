# Imports:
# --------
import os
import math
import glob
import torch
import numpy as np
from ppo import PPO
import scienceplots
import matplotlib.pyplot as plt


def calculate_c_max_pi(policy_distribution):
    return np.max(policy_distribution, axis=1)

def calculate_c_ratio_pi(policy_distribution):
    max_prob = np.max(policy_distribution, axis=1)
    min_prob = np.min(policy_distribution, axis=1)
    return max_prob / (1 + min_prob)

def calculate_c_delta_pi(policy_distribution):
    return np.max(policy_distribution, axis=1) - np.min(policy_distribution, axis=1)

def calculate_c_sigma_pi(policy_distribution):
    return np.std(policy_distribution, axis=1)

def calculate_c_var_pi(policy_distribution):
    return np.var(policy_distribution, axis=1)

def calculate_c_skewness_pi(policy_distribution):
    mean_prob = np.mean(policy_distribution, axis=1)
    median_prob = np.median(policy_distribution, axis=1)
    std_prob = np.std(policy_distribution, axis=1)
    skewness = (mean_prob - median_prob) / std_prob
    return 1 - (3 * skewness)

def calculate_c_h_pi(policy_distribution):
    return 1-(-np.sum(policy_distribution * np.log(policy_distribution + 1e-10), axis=1))

def get_model_paths(directory):
    return glob.glob(os.path.join(directory, '*.pth'))

def create_state_grid(min: float = 0.0,
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

def calculate_policy_metrics(policy_distributions):
    metrics = {
        'C_max_pi': calculate_c_max_pi(policy_distribution=policy_distributions),
        'C_ratio_pi': calculate_c_ratio_pi(policy_distribution=policy_distributions),
        'C_delta_pi': calculate_c_delta_pi(policy_distribution=policy_distributions),
        'C_sigma_pi': calculate_c_sigma_pi(policy_distribution=policy_distributions),
        'C_var_pi': calculate_c_var_pi(policy_distribution=policy_distributions),
        'C_skewness_pi': calculate_c_skewness_pi(policy_distribution=policy_distributions),
        'C_h_pi': calculate_c_h_pi(policy_distribution=policy_distributions),
    }
    return metrics

def plot_heatmap(data_points,
                c,
                index,
                saving_path="heatmaps",
                clip_colourbar=True,
                colourbar_vmax=0.65,
                colourbar_vmin=0):
    """
    Creates heatmaps for different criticality metrics.

    output: saves heatmaps as .PNG files
    """
    data_points = data_points.cpu().detach()

    for name, each_metric in c.items():
        # criticality = each_metric(q_values)
        plt.style.use(['science', 'ieee'])

        if clip_colourbar:
            # NOTE: You can keep the colourbar the same by clipping values
            plt.scatter(data_points[:, 0],
                        data_points[:, 1],
                        c=each_metric,
                        alpha=0.3,
                        cmap="hot_r",
                        vmin=colourbar_vmin,
                        vmax=colourbar_vmax)

        else:
            plt.scatter(data_points[:, 0],
                        data_points[:, 1],
                        c=each_metric,
                        alpha=0.3,
                        cmap="hot_r")

        plt.colorbar()
        plt.savefig(os.path.join(saving_path, f"{name}_{index}.png"), 
                    dpi=300, 
                    bbox_inches='tight')
        plt.close()

def run_criticality_evolution(model_directory, heatmaps_path, device='cpu'):
    model_paths = get_model_paths(model_directory)

    # NOTE: create_state_grid() returns values [0, 1] which are normalized
    grid = create_state_grid().to(device)

    # NOTE: Normalize the states:
    # grid = grid/10.0

    model = PPO(state_dim=2, action_dim=4).to(device)

    for model_path in sorted(model_paths, key=lambda x: int(x.split(".")[0].split("_", -1)[-1])):
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # FIXME: Single point input or batch input? 🤔
        with torch.no_grad():
            # for state in grid:
            pi_output = model.pi(grid)

            policy_distributions = pi_output.cpu().detach().numpy()

        criticality_metrics = calculate_policy_metrics(np.array(policy_distributions))

        model_name = os.path.basename(model_path).split('.')[0]

        plot_heatmap(saving_path=heatmaps_path, clip_colourbar=False, data_points=grid, c=criticality_metrics, index=model_name)


if __name__=="__main__":
    run_criticality_evolution(model_directory="/media/karpenahalli/Daten/Research/1_Interesting_States/Critical-States-in-Reinforcement-Learning-A-Survey-and-New-Perspectives/ppo_training_results/PPO_single_exp_1a", 
                              heatmaps_path="/media/karpenahalli/Daten/Research/1_Interesting_States/Critical-States-in-Reinforcement-Learning-A-Survey-and-New-Perspectives/ppo_training_results/PPO_single_exp_1a/heatmaps", 
                              device='cpu')
    