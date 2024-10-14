from dm_control import composer
from dm_control.locomotion.arenas import floors, corridors
import sys

import numpy as np
from dm_control.utils import rewards
from dm_control import mjcf

# Constants
_CONTROL_TIMESTEP = 0.02
_PHYSICS_TIMESTEP = 0.005
_SPAWN_POS = [0, 0.0, 0.02]

# Base Task class, shared across all environments
class BaseTask(composer.Task):
	def __init__(self, walker, arena, max_steps, physics_timestep, control_timestep, add_cam=False, add_ramp=False):
		self._arena = arena
		self._walker = walker
		self.max_steps = max_steps
		self.physics_timestep = physics_timestep
		self.control_timestep = control_timestep
		self.episode_rewards = []
		self._should_terminate = False
		self.prev_action = np.zeros(32)  # Track previous actions

		# Attach walker to the arena
		spawn_site = self._arena.mjcf_model.worldbody.add('site', pos=_SPAWN_POS)
		self._walker.create_root_joints(arena.attach(self._walker, spawn_site))
		spawn_site.remove()

		# Set timesteps and buffer
		self.set_timesteps(physics_timestep=self.physics_timestep, control_timestep=self.control_timestep)
		self._buffer_size = int(round(self.control_timestep / self.physics_timestep))

		# Explicitly enable observables.
		enabled_observables = []
		enabled_observables += self._walker.observables.proprioception
		enabled_observables += self._walker.observables.kinematic_sensors
		enabled_observables += self._walker.observables.dynamic_sensors
		enabled_observables.append(self._walker.observables.sensors_touch)

		for observable in enabled_observables:
			observable.enabled = True


	def before_step(self, physics, action, random_state):
		"""Apply action and update previous actions."""
		self.prev_action = action
		action = action * (physics.model.actuator_ctrlrange[:, 1] - physics.model.actuator_ctrlrange[:, 0]) / 2 + \
				 (physics.model.actuator_ctrlrange[:, 1] + physics.model.actuator_ctrlrange[:, 0]) / 2
		self._walker.apply_action(physics, action, random_state)

	def initialize_episode_mjcf(self, random_state: np.random.RandomState):
		if hasattr(self._arena, 'regenerate'):
			self._arena.regenerate(random_state) # type: ignore
			# Optionally modify things in the arena before it's compiled to physics.    
			
	def initialize_episode(self, physics, random_state):
		pass

	def get_reward(self, physics):
		return 0.0

	def should_terminate_episode(self, physics):
		return self._should_terminate

# Corridor Task (Rodent walker in a corridor environment)
class CorridorTask(BaseTask):
	def __init__(self, walker, arena, max_steps, physics_timestep, control_timestep, add_cam=False, add_ramp=False):
		super().__init__(walker, arena, max_steps, physics_timestep, control_timestep, add_cam=add_cam, add_ramp=add_ramp)

		# Custom observables for velocity and previous action
		self._walker.observables.add_observable('vel_obs', self.vel_obs)
		self._walker.observables.add_observable('prev_action', self.previous_action)

	@composer.observable
	def vel_obs(self):
		def get_vel_obs(physics: mjcf.Physics):
			xvel = physics.bind(self._walker.root_body).subtree_linvel[0]
			return np.array([xvel])
		return composer.observation.Generic(get_vel_obs)

	@composer.observable
	def previous_action(self):
		def get_prev_action(physics):
			return self.prev_action
		return composer.observation.Generic(get_prev_action)

	def initialize_episode(self, physics, random_state):
		physics.data.qpos[:2] = [0, 0]
		self.target_speed = 0.45
		physics.model.geom_rgba[1, :] = (0, 1, 0, 1)
		physics.model.geom_rgba[2, :] = (0, 1, 0, 1)
		self.episode_rewards = []
		self._should_terminate = False

	def get_reward(self, physics):
		xvel = physics.bind(self._walker.root_body).subtree_linvel[0]
		target_vel = self.target_speed
		diff_vel = np.abs(xvel - target_vel)
		vel_reward = rewards.tolerance(diff_vel, bounds=(0.0, 0.0), margin=0.2, value_at_margin=0.0, sigmoid='linear')

		y_pos = np.abs(physics.data.qpos[1])
		if y_pos > 0.2:
			self._should_terminate = True
			return -10

		y_pos_penalty = -rewards.tolerance(y_pos - 0.2, bounds=(0.0, 0.0), margin=0.15, value_at_margin=0.0, sigmoid='linear')

		reward = vel_reward + y_pos_penalty
		self.episode_rewards.append(reward)
		self.check_termination(physics)
		return reward

	def check_termination(self, physics):
		if len(self.episode_rewards) > 150 and np.mean(self.episode_rewards[-100:]) < 0.1:
			self._should_terminate = True

# Fetch Task (Rodent walker fetching objects)
class FetchTask(BaseTask):
	def __init__(self, walker, arena, max_steps, physics_timestep, control_timestep, add_cam=False, add_ramp=False):
		super().__init__(walker, arena, max_steps, physics_timestep, control_timestep, add_cam=add_cam, add_ramp=add_ramp)

		# Ball setup
		ball_radius = 0.03
		ball = self._arena.mjcf_model.worldbody.add('body', pos=(0, 0, ball_radius))
		ball.add('geom', type='sphere', size=(ball_radius,))
		self._ball_joint = ball.add('freejoint', name='ball')

		# Custom observable for ball position and previous action
		self._walker.observables.add_observable('ball_rat_obs', self.ball_rat_obs)
		self._walker.observables.add_observable('prev_action', self.previous_action)

	@composer.observable
	def ball_rat_obs(self):
		def get_ball_pos(physics: mjcf.Physics):
			nose_pos = physics.named.data.site_xpos['walker/nose_0_kpsite'][:3]
			ball_pos = physics.named.data.qpos['ball'][:3]
			return np.concatenate((nose_pos, ball_pos))
		return composer.observation.Generic(get_ball_pos)

	@composer.observable
	def previous_action(self):
		def get_prev_action(physics):
			return self.prev_action
		return composer.observation.Generic(get_prev_action)

	def initialize_episode(self, physics, random_state):
		physics.data.qvel[:-7] = 0
		physics.data.qpos[:-7] = np.random.uniform(-0.5, 0.5, size=physics.data.qpos[:-7].shape)
		self.episode_rewards = []
		self._should_terminate = False

	def get_reward(self, physics):
		ball_pos = physics.named.data.qpos['ball'][:2]
		nose_pos = physics.named.data.site_xpos['walker/nose_0_kpsite'][:2]
		distance = np.linalg.norm(nose_pos - ball_pos)

		# Reward calculation based on distance to ball
		reward = -distance
		if distance < 0.05:
			reward = 10
		if nose_pos[0] > 0.8 or nose_pos[0] < -0.8 or nose_pos[1] > 0.8 or nose_pos[1] < -0.8:
			reward = -5

		self.episode_rewards.append(reward)
		self.check_termination(physics)
		return reward

	def check_termination(self, physics):
		if self.episode_rewards[-1] == 10 or self.episode_rewards[-1] == -5:
			self._should_terminate = True

# Path Task (Rodent walker following a trajectory)
class PathTask(BaseTask):
	def __init__(self, walker, arena, max_steps, physics_timestep, control_timestep, dataset_dir, training_data, add_cam=False, add_ramp=False):
		super().__init__(walker, arena, max_steps, physics_timestep, control_timestep, add_cam=add_cam, add_ramp=add_ramp)

		self.training_data = training_data  # Loaded trajectory data

		# Custom observable for future positions and previous action
		self._walker.observables.add_observable('future_xpos', self.future_xpos)
		self._walker.observables.add_observable('prev_action', self.previous_action)

	@composer.observable
	def future_xpos(self):
		def get_future_xpos(physics):
			step = round(physics.time() / self.control_timestep)
			return self.training_data[step][:2]
		return composer.observation.Generic(get_future_xpos)

	@composer.observable
	def previous_action(self):
		def get_prev_action(physics):
			return self.prev_action
		return composer.observation.Generic(get_prev_action)

	def initialize_episode(self, physics, random_state):
		physics.data.qvel[:] = 0
		physics.data.qpos[:] = self.training_data[0]
		self.episode_rewards = []
		self._should_terminate = False

	def get_reward(self, physics):
		step = round(physics.time() / self.control_timestep)
		rat_pos = physics.data.qpos[:2]
		path_pos = self.training_data[step][:2]
		distance = np.linalg.norm(rat_pos - path_pos)
		distance_rwrd = rewards.tolerance(distance, bounds=(0, 0), sigmoid='linear', margin=0.15, value_at_margin=0.0)

		# Quaternion difference for orientation matching
		rat_quat = physics.data.qpos[3:7]
		path_quat = self.training_data[step][3:7]
		diff = np.arccos(2 * np.sum(rat_quat * path_quat) ** 2 - 1)
		quat_rwrd = rewards.tolerance(diff, bounds=(0, 0), sigmoid='linear', margin=0.5, value_at_margin=0.0)

		reward = 0.5 * distance_rwrd + 0.5 * quat_rwrd
		self.episode_rewards.append(reward)
		self.check_termination(step)
		return reward

	def check_termination(self, step):
		if step > len(self.training_data) - 85:
			self._should_terminate = True


# Unified environment function
def rodent_env(task_type, random_state=None, max_steps=500, dataset_dir=None, training_data=None, add_cam=False, add_ramp=False):
	walker = rodent.Rat()
	
	if task_type == "corridor":
		arena = corridors.EmptyCorridor(corridor_width=0.5)
		task = CorridorTask(walker, arena, max_steps, physics_timestep=_PHYSICS_TIMESTEP, control_timestep=_CONTROL_TIMESTEP, add_cam=add_cam, add_ramp=add_ramp)
	elif task_type == "fetch":
		arena = floors.Floor(size=(0.762, 0.762))
		task = FetchTask(walker, arena, max_steps, physics_timestep=_PHYSICS_TIMESTEP, control_timestep=_CONTROL_TIMESTEP, add_cam=add_cam, add_ramp=add_ramp)
	elif task_type == "path":
		arena = floors.Floor(size=(0.762, 0.762))
		task = PathTask(walker, arena, max_steps, physics_timestep=_PHYSICS_TIMESTEP, control_timestep=_CONTROL_TIMESTEP, dataset_dir=dataset_dir, training_data=training_data,)

	return composer.Environment(time_limit=max_steps * _CONTROL_TIMESTEP, task=task, random_state=random_state)
