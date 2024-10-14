
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

from stage1_rodent_task import rodent_env,_CONTROL_TIMESTEP



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
    parser.add_argument("--total-timesteps", type=int, default=400000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float,default=2e-5,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=64,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=180,
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
    

    parser.add_argument("-vaebeta","--vaebeta", type=float, default=1.0,
        help="beta for vae loss")
    
    
    parser.add_argument("--xpos-std", type=int, default=170,
        help="std of xpos")


    parser.add_argument("--print-every", type=int, default=100,
        help="print and store training reward")
    parser.add_argument("-ld", "--load", action="store_true", help="load a trained model")
    parser.add_argument("-test", "--test", action="store_true", help="only test run")



    parser.add_argument(
    "-smd",
    "--save-model-dir",
    default=project_folder+"/trained_models/",
    help="folder to save trained models",
    )
    parser.add_argument(
    "-svd",
    "--save-video-dir",
    default=project_folder+"/save_videos/",
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



    parser.add_argument("--loadmodel", type=int, default=831, help=" number of load model")
    parser.add_argument(
    "-lmd",
    "--load-model-dir",
    default=project_folder+"/trained_models/",
    help="folder to save trained models",
    )



    parser.add_argument("--folder-number", type=int, default=0, help="clip number to train upon")
    parser.add_argument("-clipnum","--clip-train-number",type=int, default=-1, 
                        help="clip number to train upon if its -1 then all clips are used")



    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return  args

def get_init_configs(args):
  
    real_arena_x = 1.524/2
    xaxis_scale_factor = 1
    real_arena_y = 1.524/2
    yaxis_scale_factor = 1
    mujoco_arena_x = real_arena_x*xaxis_scale_factor
    mujoco_arena_y= real_arena_y*yaxis_scale_factor

    
    std = {'com': 1200, 'qvel': 0.0008, 'root2site': 750, 'joint_quat': 0.45,'site_xpos':args.xpos_std}
    weights = (0,0,0,0,1)

    physics_timestep = float(_CONTROL_TIMESTEP)/20
    
    site_weights = np.ones((20,1))
    site_weights[9][0] = 0
    site_weights[13][0] = 0

    datasetname =  "imitation_train_data.pkl"

    return (mujoco_arena_x,mujoco_arena_y,std,weights,physics_timestep,site_weights,datasetname)


class CustomObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(353,), dtype=np.float32)

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
               
            keys.remove('walker/future_xpos')
            keys.remove('walker/current_step')
            keys.remove('walker/prev_action')

            observation_arrays = [observation[key].ravel() for key in keys]
            return type(observation)([(output_key, np.concatenate(observation_arrays))])

        current_step = np.array([ob['walker/current_step']])
        future_xpos  = ob['walker/future_xpos']
        prev_action = ob['walker/prev_action']
        ob = flatten_observation(ob)[FLAT_OBSERVATION_KEY]
        ob = np.concatenate((current_step,future_xpos,ob,prev_action))
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

def make_env(rank,args,clip_ids_to_use):
    def thunk():

        (mujoco_arena_x,mujoco_arena_y,std,weights,physics_timestep,site_weights,datasetname)= get_init_configs(args)
        env = rodent_env(floor_size=(mujoco_arena_x,mujoco_arena_y),std=std,
                         weights=weights,physics_timestep=physics_timestep,rank=rank,
                         site_weights=site_weights,
                         dataset_dir=args.dataset_dir,
                         datasetname=datasetname,
                         clip_ids_to_use=clip_ids_to_use
                        )
    
        env = DmControlCompatibilityV0(env,render_mode = "rgb_array",render_kwargs = {"camera_id":3,"width":640,"height":480})
        trigger = lambda t: t % args.print_every == 0

        if args.clip_train_number==-1:
            env = gym.experimental.wrappers.RecordVideoV0(env, video_folder= args.save_video_dir+str(rank), episode_trigger=trigger, disable_logger=True,fps=15)
        else:
            if rank==0:
                env = gym.experimental.wrappers.RecordVideoV0(env, video_folder= args.save_video_dir+str(rank), episode_trigger=trigger, disable_logger=True,fps=15)
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

        self.xpos_conv = nn.Conv2d(1, 64, (1,60), stride=(1,60))
        self.xpos_linear1 = layer_init(nn.Linear(64*3, 100))
        self.xpos_linear2 = layer_init(nn.Linear(100, 100))

        self.hidden2latent_mean = nn.Linear(100, 400)  # Mean of latent distribution
        self.hidden2latent_logvar = nn.Linear(100, 400)  # Log variance of latent distribution

    def reparameterize(self, mean, logvar):
        # Reparameterization trick to sample from N(mean, var) using N(0, 1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
        

    def forward(self, obs):

        xpos_obs = obs[:,1:181]
        xpos_obs = torch.unsqueeze(xpos_obs,1)
        xpos_obs = torch.unsqueeze(xpos_obs,1)
        xpos_layer = F.relu(self.xpos_conv(xpos_obs))
        xpos_layer = xpos_layer.view(xpos_layer.size(0), -1)

        hidden1 = F.relu(self.xpos_linear1(xpos_layer))
        hidden2 = F.tanh(self.xpos_linear2(hidden1))

        # Obtain mean and log variance for latent space
        latent_mean = self.hidden2latent_mean(hidden2)
        latent_logvar = self.hidden2latent_logvar(hidden2)

        z = self.reparameterize(latent_mean, latent_logvar)


        return z, latent_mean, latent_logvar



    
class LLC(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.prop_layers = nn.Sequential(
            layer_init(nn.Linear(140, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
        )

        self.prev_action_layer = layer_init(nn.Linear(32, 32))

        self.lstm = nn.LSTM(128+32+400, 128)

        self.combine_layer2 = layer_init(nn.Linear(128, 128))

        self.actor_mean = layer_init(nn.Linear(128, np.prod(envs.single_action_space.shape)), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))
        self.critic = layer_init(nn.Linear(128, 1), std=1)

        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
    
    def get_states(self, obs, lstm_state, done, hlc_hidden):
        prop_obs = obs[:,181:181+140]
        prev_action = obs[:,181+140:]
        prop_layer = self.prop_layers(prop_obs)
        prev_action_layer = F.tanh(self.prev_action_layer(prev_action))
        current_and_future = torch.cat((prop_layer,prev_action_layer,hlc_hidden),dim=1)

        

        batch_size = lstm_state[0].shape[1]
        llc_hidden = current_and_future.reshape((-1, batch_size, self.lstm.input_size))

        done = done.reshape((-1, batch_size))
        new_llc_hidden = []
        for h, d in zip(llc_hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_llc_hidden += [h]
        new_llc_hidden = torch.flatten(torch.cat(new_llc_hidden), 0, 1)

        llc_hidden = F.relu(self.combine_layer2(new_llc_hidden))
        return llc_hidden, lstm_state

        

    def get_value(self, obs, lstm_state, done, hlc_hidden):
        llc_hidden, lstm_state = self.get_states(obs, lstm_state, done, hlc_hidden)
        return self.critic(llc_hidden)



    def get_action_and_value(self, obs, lstm_state, done, hlc_hidden,action=None):

        llc_hidden, lstm_state = self.get_states(obs, lstm_state, done, hlc_hidden)
    

        action_mean = self.actor_mean(llc_hidden)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(llc_hidden), lstm_state


def compute_kl_divergence(mean, logvar):
    # Compute the KL divergence between the latent distribution and standard normal N(0, I)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return kl_divergence


if __name__ == "__main__":
    args = parse_args()


    all_clip_lengths = np.load(args.dataset_dir+"all_lengths_final.npy")
    all_clip_lengths = all_clip_lengths[:,1] - all_clip_lengths[:,2]-21+10

    clip_len_threshold = 15
    mask = all_clip_lengths>=clip_len_threshold
    clip_ids_to_use = np.where(mask)[0]
    all_clip_lengths = all_clip_lengths[mask]


    if args.clip_train_number==-1:
        
        if args.num_envs < len(all_clip_lengths):
            all_clip_lengths = all_clip_lengths[:args.num_envs]
            clip_ids_to_use = clip_ids_to_use[:args.num_envs]
            print("Number of clips is more than number of envs, reducing number of clips to number of envs")
    else:
        clip_ids_to_use = [clip_ids_to_use[args.clip_train_number]]
        all_clip_lengths = [all_clip_lengths[args.clip_train_number]]



    print("Number of clips to use: ",len(all_clip_lengths))

    
    args.save_model_dir = args.save_model_dir+str(args.folder_number)+'/'
    print(args.save_model_dir)
    args.save_video_dir = args.save_video_dir+str(args.folder_number)+'/'
    if not os.path.exists(args.save_model_dir):
        os.makedirs(args.save_model_dir)
    if not os.path.exists(args.save_video_dir):
        os.makedirs(args.save_video_dir)
    
    if args.clip_train_number==-1:
        for i in range(args.num_envs):
            if not os.path.exists(args.save_video_dir+str(i)):
                os.makedirs(args.save_video_dir+str(i))
    else:
        if not os.path.exists(args.save_video_dir+'0'):
            os.makedirs(args.save_video_dir+'0')

    run_name = f"{int(time.time())}_{args.folder_number}"
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

    # env setup
    envs = gym.vector.AsyncVectorEnv(
        [make_env(i,args,clip_ids_to_use) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    envs = gym.wrappers.NormalizeObservation(envs)

    hlc = HLC().to(device)
    llc = LLC().to(device)
    

    if args.load:
        args.load_model_dir = args.load_model_dir+str(args.loadmodel)+'/'

        print("Loading models from: ",args.load_model_dir)

        hlc.load_state_dict(torch.load(f"{args.load_model_dir}best_hlc.dat"))
        llc.load_state_dict(torch.load(f"{args.load_model_dir}best_llc.dat"))
        with open(f"{args.load_model_dir}rms.pkl", 'rb') as f:
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


    next_lstm_state = (
        torch.zeros(llc.lstm.num_layers, args.num_envs, llc.lstm.hidden_size).to(device),
        torch.zeros(llc.lstm.num_layers, args.num_envs, llc.lstm.hidden_size).to(device),
    )  # hidden and cell states (see https://youtu.be/8HyCNIVRbSU)

    episode_sum_rewards = np.zeros(len(all_clip_lengths))
    store_all_episode_sum_rewards = [[] for i in range(len(all_clip_lengths))]
    episode_sum_lengths = np.zeros(len(all_clip_lengths))
    store_all_episode_sum_lengths = [[] for i in range(len(all_clip_lengths))]
    last_N_normalised_rewards = np.zeros(len(all_clip_lengths))
    last_N_normalised_lengths = np.zeros(len(all_clip_lengths))
    max_reward = -1000000

    store_hidden = np.zeros((0, 400))

   
    for update in range(1, num_updates + 1):
        initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())
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
                hidden,_,_ = hlc(next_obs)
                action, logprob, _, value,next_lstm_state = llc.get_action_and_value(next_obs,next_lstm_state,next_done,hidden)
                #action = torch.clamp(action,-1,1)
                values[step] = value.flatten()

            if args.test:
                store_hidden = np.concatenate((store_hidden, hidden.cpu().numpy()), axis=0)

            
            
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
                     
                        writer.add_scalar(f"mean_reward_clip_{clip_ids_to_use[clip_num]}_len_{all_clip_lengths[clip_num]}", 
                                        round(mean_reward/all_clip_lengths[clip_num]*100,1), 
                                        (len(store_all_episode_sum_rewards[clip_num])+1))
                        writer.add_scalar(f"mean_length_clip_{clip_ids_to_use[clip_num]}_len_{all_clip_lengths[clip_num]}", 
                                        round(mean_length/all_clip_lengths[clip_num]*100,1),
                                        (len(store_all_episode_sum_rewards[clip_num])+1))
        if update%args.print_every==0:
            current_avg_reward = np.mean(last_N_normalised_rewards)
            current_avg_length = np.mean(last_N_normalised_lengths)

            if args.clip_train_number==-1:
                writer.add_scalar("avg_reward", current_avg_reward, update)
                writer.add_scalar("avg_length", current_avg_length, update)

            if current_avg_reward > max_reward: 
                max_reward = current_avg_reward
                torch.save(hlc.state_dict(), f"{args.save_model_dir}best_hlc.dat")
                torch.save(llc.state_dict(), f"{args.save_model_dir}best_llc.dat")
                with open(f"{args.save_model_dir}rms.pkl", 'wb') as f:
                    pickle.dump(envs.obs_rms, f)
                    
        
                    
        if not args.test:
            # bootstrap value if not done
            with torch.no_grad():
                hidden,_,_ = hlc(next_obs)
                next_value = llc.get_value(next_obs,next_lstm_state,next_done,hidden).reshape(1, -1)
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

            b_dones = dones.reshape(-1)

            # Optimizing the policy and value network
         

            assert args.num_envs % args.num_minibatches == 0
            envsperbatch = args.num_envs // args.num_minibatches
            envinds = np.arange(args.num_envs)
            flatinds = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)



            clipfracs = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(envinds)
                for start in range(0, args.num_envs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mb_inds = flatinds[:, mbenvinds].ravel()  # be really careful about the index

                    hidden,latent_mean, latent_logvar = hlc(b_obs[mb_inds])
                    _, newlogprob, entropy, newvalue, _ = llc.get_action_and_value(
                    b_obs[mb_inds],
                    (initial_lstm_state[0][:, mbenvinds], initial_lstm_state[1][:, mbenvinds]),
                    b_dones[mb_inds],
                    hlc_hidden=hidden,
                    action=b_actions[mb_inds]
                    )
                    
                    
                    
                    
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

                    kl_divergence = compute_kl_divergence(latent_mean, latent_logvar)


                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + args.vaebeta * kl_divergence

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(list(hlc.parameters())+list(llc.parameters()), args.max_grad_norm)
                    optimizer.step()

                if args.target_kl is not None:
                    if approx_kl > args.target_kl:
                        break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    if args.test:
        np.save(args.save_model_dir+"hidden.npy",store_hidden)
        envs.close()

    envs.close()
    
