# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import argparse
import os
import csv 
import warnings ; warnings.warn = lambda *args,**kwargs: None

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ["MUJOCO_GL"] = "osmesa"

import sys
from os.path import dirname, abspath
project_folder = dirname(dirname(abspath(__file__)))
sys.path.insert(1, project_folder)


import random
import time
from distutils.util import strtobool
import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium import ActionWrapper
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import collections
from shimmy.dm_control_compatibility import DmControlCompatibilityV0

import pickle
# The basic mujoco wrapper.
from dm_control import mujoco

# Access to enums and MuJoCo library functions.
from dm_control.mujoco.wrapper.mjbindings import enums
from dm_control.mujoco.wrapper.mjbindings import mjlib

# PyMJCF
from dm_control import mjcf
import torch.nn.functional as F


# Composer high level imports
from dm_control import composer
from dm_control.composer.observation import observable
from dm_control.composer import variation

# Imports for Composer tutorial example
from dm_control.composer.variation import distributions
from dm_control.composer.variation import noises
from dm_control.locomotion.arenas import floors

# Control Suite
from dm_control import suite

# Run through corridor example
from dm_control.locomotion.walkers import cmu_humanoid
from dm_control.locomotion.arenas import corridors as corridor_arenas
from dm_control.locomotion.tasks import corridors as corridor_tasks

from stage3_box_task import box_env,_CONTROL_TIMESTEP



def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=16,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="fetchball",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=150000000,
        help="total timesteps of the experiments")
    parser.add_argument('-lr',"--learning-rate", type=float,default=1e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=64,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=100,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=1,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=10,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")


    parser.add_argument("--print-every", type=int, default=50,
        help="print and store training reward")
    parser.add_argument("-ld", "--load", action="store_true", help="load a trained model")
    parser.add_argument("-test", "--test", action="store_true", help="only test run")


    parser.add_argument(
    "-smd",
    "--save-model-dir",
    default=project_folder+"/stage_3_trained/",
    help="folder to save trained models",
    )
    parser.add_argument(
    "-llcepoch",
    "--load-llc-epoch", 
    type=int, 
    default=1000, 
    help="llc load epoch")

    parser.add_argument(
    "-svd",
    "--save-video-dir",
    default=project_folder+"/save_videos_stage3/",
    help="folder to save trained models",
    )
    parser.add_argument(
    "-dr",
    "--dataset-dir",
    default=project_folder+"/dataset/",
    help="folder to save trained models",
    )
    parser.add_argument(
    "-rd",
    "--runs-dir",
    default=project_folder+"/runs/",
    help="folder to save trained models",
    )
    parser.add_argument(
    "-maxepilength",
    "--max-episode-length", 
    type=int, 
    default=50, 
    )

    parser.add_argument('-fn',"--folder-number", type=int, default=0, help="clip number to train upon")


    parser.add_argument("-rewardnormalise", "--rewardnormalise", action="store_true")



    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return  args

def get_init_configs(args):
  
    real_arena_x =0.762
    xaxis_scale_factor = 1
    real_arena_y = 0.762
    yaxis_scale_factor = 1
    mujoco_arena_x = real_arena_x*xaxis_scale_factor
    mujoco_arena_y= real_arena_y*yaxis_scale_factor

    

    physics_timestep = float(_CONTROL_TIMESTEP)/20


    return (mujoco_arena_x,mujoco_arena_y,physics_timestep)


class CustomObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(29,), dtype=np.float32)

    def observation(self, ob):
        FLAT_OBSERVATION_KEY = 'observations'
        
        def flatten_observation(observation, output_key=FLAT_OBSERVATION_KEY):

            if not isinstance(observation, collections.abc.MutableMapping):
                raise ValueError('Can only flatten dict-like observations.')

            if isinstance(observation, collections.OrderedDict):
                keys = observation.keys()
            else:
                # Keep a consistent ordering for other mappings.
                keys = sorted(observation.keys())

            keys.remove('walker/ball_rat_obs')
            
            observation_arrays = [observation[key].ravel() for key in keys]
            return type(observation)([(output_key, np.concatenate(observation_arrays))])

        ball_rat_pos  = ob['walker/ball_rat_obs']
        ob = flatten_observation(ob)[FLAT_OBSERVATION_KEY]
        ob = np.concatenate((ball_rat_pos,ob))
    
        return ob   



class ClipAction(ActionWrapper):
    def __init__(self, env: gym.Env):
        """A wrapper for clipping continuous actions within the valid bound.

        Args:
            env: The environment to apply the wrapper
        """
        assert isinstance(env.action_space, Box)
        super().__init__(env)

    def action(self, action):
        """Clips the action within the valid bounds.

        Args:
            action: The action to clip

        Returns:
            The clipped action
        """
        return np.clip(action, -1, 1)

def make_env(rank,args):
    def thunk():

        (mujoco_arena_x,mujoco_arena_y,physics_timestep)= get_init_configs(args)
        env = box_env(floor_size=(mujoco_arena_x,mujoco_arena_y),
                         physics_timestep=physics_timestep,
                        max_steps=args.max_episode_length,
                        )
        if not args.test:
            env = DmControlCompatibilityV0(env,render_mode = "rgb_array",render_kwargs = {"camera_id":1,"width":640,"height":480})
            trigger = lambda t: t % args.print_every == 0
            if rank==0:
                env = gym.experimental.wrappers.RecordVideoV0(env, video_folder= args.save_video_dir, episode_trigger=trigger, disable_logger=True,fps=5)
        else:
            env = DmControlCompatibilityV0(env,render_mode = "rgb_array",render_kwargs = {"camera_id":3,"width":1000,"height":800})
            trigger = lambda t: t % args.print_every == 0
            if rank==0:
                env = gym.experimental.wrappers.RecordVideoV0(env, video_folder= args.save_video_dir,  disable_logger=True, video_length =100,fps=8)

            
        
        
        env = CustomObservationWrapper(env)
        env = ClipAction(env)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer




class HLC(nn.Module):
    def __init__(self):
        super().__init__()

        self.hlc_linear1 = layer_init(nn.Linear(13, 8))
        self.hlc_linear2 = layer_init(nn.Linear(8, 8))

    def forward(self, obs):
        highlevelobs = obs[:,:13]
        hidden1 = F.relu(self.hlc_linear1(highlevelobs))
        hidden2 = F.tanh(self.hlc_linear2(hidden1))
        return hidden2
    
class LLC(nn.Module):
    def __init__(self):
        super().__init__()

        self.early1 = layer_init(nn.Linear(16, 8))
        self.early2 = layer_init(nn.Linear(8, 8))
        self.early3 = layer_init(nn.Linear(8, 8))
        
        self.combine_layer = layer_init(nn.Linear(8+8, 8))
        self.combine_layer2 = layer_init(nn.Linear(8, 8))

        self.llc_mean = layer_init(nn.Linear(8, 2), std=0.01)

        self.actor_logstd = nn.Parameter(torch.zeros(1, 2))
        self.critic = nn.Sequential(
            layer_init(nn.Linear(8, 8)),
            nn.Tanh(),
            layer_init(nn.Linear(8, 1)),
        )

    def get_value(self, obs, hidden):
        current_obs = obs[:,13:]
        early1 = F.relu(self.early1(current_obs))
        early2 = F.relu(self.early2(early1))
        early3 = F.relu(self.early3(early2))
        current_layer = early3

        current_and_future = torch.cat((current_layer,hidden),dim=1)
        llc_hidden1 = F.relu(self.combine_layer(current_and_future))
        llc_hidden2 = F.relu(self.combine_layer2(llc_hidden1))

        return self.critic(llc_hidden2)


    def get_action_and_value(self, obs, hidden, action=None):
        current_obs = obs[:,13:]
        early1 = F.relu(self.early1(current_obs))
        early2 = F.relu(self.early2(early1))
        early3 = F.relu(self.early3(early2))
        current_layer = early3

        current_and_future = torch.cat((current_layer,hidden),dim=1)
        llc_hidden1 = F.relu(self.combine_layer(current_and_future))
        llc_hidden2 = F.relu(self.combine_layer2(llc_hidden1))

        action_mean = self.llc_mean(llc_hidden2)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(llc_hidden2)


    

if __name__ == "__main__":
    args = parse_args()

    args.save_model_dir = args.save_model_dir+str(args.folder_number)+'/'
    print(args.save_model_dir)
    args.save_video_dir = args.save_video_dir+str(args.folder_number)+'/'
    print(args.save_video_dir)
    if not os.path.exists(args.save_model_dir):
        os.makedirs(args.save_model_dir)
    if not os.path.exists(args.save_video_dir):
        os.makedirs(args.save_video_dir)

    all_clip_lengths = [args.max_episode_length]    

    run_name = f"{int(time.time())}_{args.folder_number}_stage3"
    writer = SummaryWriter(args.runs_dir+run_name)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    if args.test:
        args.num_envs = 1
        args.load = True
        args.print_every = 1
  
    # env setup
    envs = gym.vector.AsyncVectorEnv(
        [make_env(i,args) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    envs = gym.wrappers.NormalizeObservation(envs)
    if args.rewardnormalise:
        envs  = gym.wrappers.NormalizeReward(envs)
    

        


    hlc = HLC().to(device)
    llc = LLC().to(device)

    if args.load:
        hlc.load_state_dict(torch.load(f"{args.save_model_dir}best_hlc.dat"))
        llc.load_state_dict(torch.load(f"{args.save_model_dir}best_llc.dat"))
        with open(f"{args.save_model_dir}rms.pkl", 'rb') as f:
            envs.obs_rms = pickle.load(f)


    

    optimizer = optim.Adam(list(hlc.parameters())+list(llc.parameters()), lr=args.learning_rate, eps=1e-5)



    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    episode_sum_rewards = np.zeros(len(all_clip_lengths))
    store_all_episode_sum_rewards = [[] for i in range(len(all_clip_lengths))]
    episode_sum_lengths = np.zeros(len(all_clip_lengths))
    store_all_episode_sum_lengths = [[] for i in range(len(all_clip_lengths))]
    last_N_normalised_rewards = np.zeros(len(all_clip_lengths))
    last_N_normalised_lengths = np.zeros(len(all_clip_lengths))
    max_reward = -10000

   
    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                hidden = hlc(next_obs)
                action, logprob, _, value = llc.get_action_and_value(next_obs, hidden)
                #action = torch.clamp(action,-1,1)
                values[step] = value.flatten()

            
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            for clip_num in range(len(all_clip_lengths)):
                episode_sum_rewards[clip_num] += np.sum(reward[clip_num])
                episode_sum_lengths[clip_num] += 1
                if done[clip_num]:
                    store_all_episode_sum_rewards[clip_num].append(episode_sum_rewards[clip_num])
                    store_all_episode_sum_lengths[clip_num].append(episode_sum_lengths[clip_num])
                    episode_sum_rewards[clip_num] = 0
                    episode_sum_lengths[clip_num] = 0


                    if (len(store_all_episode_sum_rewards[clip_num])+1)%args.print_every==0:
                        mean_reward = np.mean(store_all_episode_sum_rewards[clip_num][-args.print_every:])
                        mean_length = np.mean(store_all_episode_sum_lengths[clip_num][-args.print_every:])

                        last_N_normalised_rewards[clip_num] = round(mean_reward/all_clip_lengths[clip_num]*100,1)
                        last_N_normalised_lengths[clip_num] = round(mean_length/all_clip_lengths[clip_num]*100,1)

                        if not args.rewardnormalise:
                            writer.add_scalar(f"mean_reward", 
                                            round(mean_reward,1), 
                                            (len(store_all_episode_sum_rewards[clip_num])+1))
                        else:
                            writer.add_scalar(f"mean_reward_normalised", 
                                            round(mean_reward,1), 
                                            (len(store_all_episode_sum_rewards[clip_num])+1))

                        
                        writer.add_scalar(f"mean_len", 
                                        round(mean_length,1), 
                                        (len(store_all_episode_sum_lengths[clip_num])+1))
                 
        if not args.test:

            if update%args.print_every==0:
                current_avg_reward = np.mean(last_N_normalised_rewards)
                current_avg_length = np.mean(last_N_normalised_lengths)

                if current_avg_reward > max_reward: 
                    max_reward = current_avg_reward
                    torch.save(hlc.state_dict(), f"{args.save_model_dir}best_hlc.dat")
                    torch.save(llc.state_dict(), f"{args.save_model_dir}best_llc.dat")
                    with open(f"{args.save_model_dir}rms.pkl", 'wb') as f:
                        pickle.dump(envs.obs_rms, f)
                        
        

        
            # bootstrap value if not done
            with torch.no_grad():
                hidden = hlc(next_obs)
                next_value = llc.get_value(next_obs,hidden).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            # flatten the batch
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    hidden = hlc(b_obs[mb_inds])
                    _, newlogprob, entropy, newvalue = llc.get_action_and_value(b_obs[mb_inds], hidden,b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(hlc.parameters(), args.max_grad_norm)
                    optimizer.step()

                if args.target_kl is not None:
                    if approx_kl > args.target_kl:
                        break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    

    envs.close()
    
