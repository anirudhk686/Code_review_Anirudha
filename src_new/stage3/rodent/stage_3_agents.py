import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Layer initialization with orthogonal weights
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# High-Level Controller for Corridor Task
class HLCCorridor(nn.Module):
    def __init__(self):
        super(HLCCorridor, self).__init__()

        self.hlc_common1 = layer_init(nn.Linear(9, 32))
        self.hlc_common2 = layer_init(nn.Linear(32, 64))
        self.hlc_common3 = layer_init(nn.Linear(64, 100))

        self.hlc_policy1 = layer_init(nn.Linear(100, 400))
        self.hidden2latent_mean = nn.Linear(400, 400)
        self.hidden2latent_logvar = nn.Linear(400, 400)

        self.hlc_critic_prop1 = layer_init(nn.Linear(140, 200))
        self.hlc_critic_prop2 = layer_init(nn.Linear(200, 100))

        self.hlc_critic1 = layer_init(nn.Linear(200, 200))
        self.hlc_critic2 = layer_init(nn.Linear(200, 1))

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, obs):
        highlevelobs = obs[:, :9]
        prop_obs = obs[:, 9:9+140]

        hlc_common = F.relu(self.hlc_common1(highlevelobs))
        hlc_common = F.relu(self.hlc_common2(hlc_common))
        hlc_common = F.relu(self.hlc_common3(hlc_common))

        hlc_policy = F.relu(self.hlc_policy1(hlc_common))
        latent_mean = self.hidden2latent_mean(hlc_policy)
        latent_logvar = self.hidden2latent_logvar(hlc_policy)
        hlc_policy = self.reparameterize(latent_mean, latent_logvar)

        hlc_critic_prop = F.relu(self.hlc_critic_prop1(prop_obs))
        hlc_critic_prop = F.relu(self.hlc_critic_prop2(hlc_critic_prop))

        hlc_critic = torch.cat((hlc_common, hlc_critic_prop), dim=1)
        hlc_critic = F.relu(self.hlc_critic1(hlc_critic))
        hlc_critic = self.hlc_critic2(hlc_critic)

        return hlc_policy, latent_mean, latent_logvar, hlc_critic


# High-Level Controller for Path Task
class HLCPath(nn.Module):
    def __init__(self):
        super(HLCPath, self).__init__()

        self.hlc_common1 = layer_init(nn.Linear(14, 128))
        self.hlc_common2 = layer_init(nn.Linear(128, 100))

        self.hlc_policy1 = layer_init(nn.Linear(100, 400))
        self.hidden2latent_mean = nn.Linear(400, 400)
        self.hidden2latent_logvar = nn.Linear(400, 400)

        self.hlc_critic_prop1 = layer_init(nn.Linear(140, 200))
        self.hlc_critic_prop2 = layer_init(nn.Linear(200, 100))

        self.hlc_critic1 = layer_init(nn.Linear(200, 200))
        self.hlc_critic2 = layer_init(nn.Linear(200, 1))

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, obs):
        highlevelobs = obs[:, :14]
        prop_obs = obs[:, 14:14+140]

        hlc_common = F.relu(self.hlc_common1(highlevelobs))
        hlc_common = F.relu(self.hlc_common2(hlc_common))

        hlc_policy = F.relu(self.hlc_policy1(hlc_common))
        latent_mean = self.hidden2latent_mean(hlc_policy)
        latent_logvar = self.hidden2latent_logvar(hlc_policy)
        hlc_policy = self.reparameterize(latent_mean, latent_logvar)

        hlc_critic_prop = F.relu(self.hlc_critic_prop1(prop_obs))
        hlc_critic_prop = F.relu(self.hlc_critic_prop2(hlc_critic_prop))

        hlc_critic = torch.cat((hlc_common, hlc_critic_prop), dim=1)
        hlc_critic = F.relu(self.hlc_critic1(hlc_critic))
        hlc_critic = self.hlc_critic2(hlc_critic)

        return hlc_policy, latent_mean, latent_logvar, hlc_critic


# High-Level Controller for Fetch Task 
class HLCFetch(nn.Module):
    def __init__(self):
        super(HLCFetch, self).__init__()

        self.hlc_common1 = layer_init(nn.Linear(13, 128))
        self.hlc_common2 = layer_init(nn.Linear(128, 100))

        self.hlc_policy1 = layer_init(nn.Linear(100, 400))
        self.hidden2latent_mean = nn.Linear(400, 400)
        self.hidden2latent_logvar = nn.Linear(400, 400)

        self.hlc_critic_prop1 = layer_init(nn.Linear(140, 200))
        self.hlc_critic_prop2 = layer_init(nn.Linear(200, 100))

        self.hlc_critic1 = layer_init(nn.Linear(200, 200))
        self.hlc_critic2 = layer_init(nn.Linear(200, 1))

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, obs):
        highlevelobs = obs[:, :13]
        prop_obs = obs[:, 13:13+140]

        hlc_common = F.relu(self.hlc_common1(highlevelobs))
        hlc_common = F.relu(self.hlc_common2(hlc_common))

        hlc_policy = F.relu(self.hlc_policy1(hlc_common))
        latent_mean = self.hidden2latent_mean(hlc_policy)
        latent_logvar = self.hidden2latent_logvar(hlc_policy)
        hlc_policy = self.reparameterize(latent_mean, latent_logvar)

        hlc_critic_prop = F.relu(self.hlc_critic_prop1(prop_obs))
        hlc_critic_prop = F.relu(self.hlc_critic_prop2(hlc_critic_prop))

        hlc_critic = torch.cat((hlc_common, hlc_critic_prop), dim=1)
        hlc_critic = F.relu(self.hlc_critic1(hlc_critic))
        hlc_critic = self.hlc_critic2(hlc_critic)

        return hlc_policy, latent_mean, latent_logvar, hlc_critic


# Low-Level Controller (LLC) - Common Across All Tasks
class LLC(nn.Module):
    def __init__(self, action_space):
        super(LLC, self).__init__()

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

        self.actor_mean = layer_init(nn.Linear(128, action_space), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_space))

        # LSTM initialization
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

    def get_states(self, obs, lstm_state, done, hlc_policy):
        prop_obs = obs[:, 9:9+140]
        prev_action = obs[:, 9+140:]
        prop_layer = self.prop_layers(prop_obs)
        prev_action_layer = F.tanh(self.prev_action_layer(prev_action))
        current_and_future = torch.cat((prop_layer, prev_action_layer, hlc_policy), dim=1)

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

    def get_value(self, hlc_critic):
        return hlc_critic

    def get_action_and_value(self, obs, lstm_state, done, hlc_policy, hlc_critic, action=None):
        llc_hidden, lstm_state = self.get_states(obs, lstm_state, done, hlc_policy)
        return llc_hidden, lstm_state


# Generalized function to select the appropriate agent based on environment type
def select_agent(env, task_type, action_space):
    if task_type == "corridor":
        return HLCCorridor(), LLC(action_space)
    elif task_type == "path":
        return HLCPath(), LLC(action_space)
    elif task_type == "fetch":
        return HLCFetch(), LLC(action_space)
    else:
        raise ValueError(f"Unknown task type: {task_type}")