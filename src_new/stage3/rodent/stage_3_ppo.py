import argparse
import os
import gymnasium as gym
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from stage_3_envs import rodent_env  # Import the unified environment
from stage_3_agents import select_agent  # Import the new agent class
from stage_3_custom_obs import select_custom_observation_wrapper  # Import the custom observation logic
from stage_3_utils import get_init_configs # Import common utilities # TODO : Import other common utilities

# Parse arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="corridor", help="Choose the task: corridor, path, or fetch")
    parser.add_argument("--total-timesteps", type=int, default=150000000, help="Total timesteps for training")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num-envs", type=int, default=48, help="Number of parallel environments")
    parser.add_argument("--seed", type=int, default=42, help="Seed for randomness")
    parser.add_argument("--print-every", type=int, default=1000, help="Print every N steps")
    parser.add_argument("--save-model-dir", type=str, default="./saved_models/", help="Directory to save models")
    parser.add_argument("--cuda", action="store_true", help="Enable CUDA")
    

	# TODO : Add more arguments here
    
    args = parser.parse_args()
    return args

# Create environment based on task type
def create_env(task_type, rank, args):
    def _thunk():
        env = rodent_env(task_type)  # Create the environment using the unified environment file
        env = select_custom_observation_wrapper(env, task_type)  # Apply the correct custom observation wrapper
        env = gym.wrappers.NormalizeObservation(env)
        return env
    return _thunk

# PPO training loop
def train_ppo():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    # Create environments
    envs = gym.vector.AsyncVectorEnv([create_env(args.env_id, i, args) for i in range(args.num_envs)])
    obs_space = envs.single_observation_space.shape[0]
    action_space = envs.single_action_space.shape[0]

    # Initialize the agent (which selects the correct HLC and LLC based on task type)
    hlc, llc =  select_agent(args.env_id, action_space)
    
	# TODO : common PPO logic here
    
    

if __name__ == "__main__":
    train_ppo()
