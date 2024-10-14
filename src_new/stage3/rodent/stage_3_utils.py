import gymnasium as gym
from gymnasium import ActionWrapper
import numpy as np
from gymnasium.spaces import Box



def get_init_configs(args):
  
	real_arena_x = 1.524/2
	xaxis_scale_factor = 1
	real_arena_y = 1.524/2
	yaxis_scale_factor = 1
	mujoco_arena_x = real_arena_x*xaxis_scale_factor
	mujoco_arena_y= real_arena_y*yaxis_scale_factor

	
	physics_timestep = 0.005

	return (mujoco_arena_x,mujoco_arena_y,physics_timestep)


class RunningMeanStd:
	"""Tracks the mean, variance and count of values."""

	# https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
	def __init__(self, epsilon=1e-4, shape=()):
		"""Tracks the mean, variance and count of values."""
		self.mean = np.zeros(shape, "float64")
		self.var = np.ones(shape, "float64")
		self.count = epsilon

	def update(self, x):
		"""Updates the mean, var and count from a batch of samples."""
		batch_mean = np.mean(x, axis=0)
		batch_var = np.var(x, axis=0)
		batch_count = x.shape[0]
		self.update_from_moments(batch_mean, batch_var, batch_count)

	def update_from_moments(self, batch_mean, batch_var, batch_count):
		"""Updates from batch mean, variance and count moments."""
		self.mean, self.var, self.count = update_mean_var_count_from_moments(
			self.mean, self.var, self.count, batch_mean, batch_var, batch_count
		)


def update_mean_var_count_from_moments(
	mean, var, count, batch_mean, batch_var, batch_count
):
	"""Updates the mean, var and count using the previous mean, var, count and batch values."""
	delta = batch_mean - mean
	tot_count = count + batch_count

	new_mean = mean + delta * batch_count / tot_count
	m_a = var * count
	m_b = batch_var * batch_count
	M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
	new_var = M2 / tot_count
	new_count = tot_count

	return new_mean, new_var, new_count


class CustomNormalizeObservation(gym.Wrapper, gym.utils.RecordConstructorArgs):
	"""This wrapper will normalize observations s.t. each coordinate is centered with unit variance.

	Note:
		The normalization depends on past trajectories and observations will not be normalized correctly if the wrapper was
		newly instantiated or the policy was changed recently.
	"""

	def __init__(self, env: gym.Env, epsilon: float = 1e-8, oldobs_rms=None):
		"""This wrapper will normalize observations s.t. each coordinate is centered with unit variance.

		Args:
			env (Env): The environment to apply the wrapper
			epsilon: A stability parameter that is used when scaling the observations.
		"""
		gym.utils.RecordConstructorArgs.__init__(self, epsilon=epsilon)
		gym.Wrapper.__init__(self, env)

		try:
			self.num_envs = self.get_wrapper_attr("num_envs")
			self.is_vector_env = self.get_wrapper_attr("is_vector_env")
		except AttributeError:
			self.num_envs = 1
			self.is_vector_env = False

		if self.is_vector_env:
			self.obs_rms = RunningMeanStd(shape=self.single_observation_space.shape)
			self.oldobs_rms = RunningMeanStd(shape=self.single_observation_space.shape)
			self.oldobs_rms.mean[13:] = oldobs_rms.mean[181:]
			self.oldobs_rms.var[13:] = oldobs_rms.var[181:]
			
		self.epsilon = epsilon

	def step(self, action):
		"""Steps through the environment and normalizes the observation."""
		obs, rews, terminateds, truncateds, infos = self.env.step(action)
		if self.is_vector_env:
			obs = self.normalize(obs)
		else:
			obs = self.normalize(np.array([obs]))[0]
		return obs, rews, terminateds, truncateds, infos

	def reset(self, **kwargs):
		"""Resets the environment and normalizes the observation."""
		obs, info = self.env.reset(**kwargs)

		if self.is_vector_env:
			return self.normalize(obs), info
		else:
			return self.normalize(np.array([obs]))[0], info

	def normalize(self, obs):
		"""Normalises the observation using the running mean and variance of the observations."""
		self.obs_rms.update(obs)
		new_rms_obs  = (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)
		old_rms_obs = (obs - self.oldobs_rms.mean) / np.sqrt(self.oldobs_rms.var + self.epsilon)
		return np.concatenate((new_rms_obs[:,:13],old_rms_obs[:,13:]),axis=1)


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
