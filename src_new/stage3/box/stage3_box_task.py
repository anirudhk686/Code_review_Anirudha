

from typing import Dict,Callable, Union, Sequence

import numpy as np
import pickle 
from dm_control import mujoco

import sys
from os.path import dirname, abspath

from dm_control.mujoco.wrapper import mjbindings
mjlib = mjbindings.mjlib
project_folder = dirname(dirname(abspath(__file__)))
sys.path.insert(1, project_folder)


from dm_control import composer
from dm_control.composer.observation import observable
from dm_control.locomotion.arenas import floors
import quaternions
from dm_control import mjcf
from dm_control.utils import rewards
from box_walker import BoxWalker

_CONTROL_TIMESTEP = 0.02
_PHYSICS_TIMESTEP = .005

_SPAWN_POS = [0,  0.0,  0.02]


def box_env(random_state=None,floor_size = (0.762, 0.762),
				physics_timestep =  _PHYSICS_TIMESTEP,
			 dataset_dir=project_folder+"/dataset/", max_steps = 150):
		"""Create simple environment."""



		global _SPAWN_POS
		global _PHYSICS_TIMESTEP
		_PHYSICS_TIMESTEP = physics_timestep


		
		walker = BoxWalker(
			physics_timestep=_PHYSICS_TIMESTEP,
			control_timestep=_CONTROL_TIMESTEP,
		)
		arena = floors.Floor(size=floor_size)
		task = RodentTask(
						walker=walker,
						arena=arena,
						physics_timestep=_PHYSICS_TIMESTEP,
						control_timestep=_CONTROL_TIMESTEP,
						dataset_dir = dataset_dir,
						max_steps = max_steps
						)
		return composer.Environment(time_limit= (max_steps*_CONTROL_TIMESTEP) ,
									task=task,
									random_state=random_state,
									strip_singleton_obs_buffer_dim=True)


class RodentTask(composer.Task):
	"""Simple rodent task."""

	def __init__(
			self,
			walker: Union['base.Walker', Callable],
			arena: composer.Arena,
			physics_timestep: float,
			control_timestep: float,
			dataset_dir,
			max_steps
	):

		self._arena = arena
		self._walker = walker
		self.dataset_dir = dataset_dir
		self.max_steps = max_steps
		self.ctrl_ranges = None
		self._should_terminate = False
		self.episode_rewards = []

	
			
		spawn_site = self._arena.mjcf_model.worldbody.add('site', pos=_SPAWN_POS)
		self._arena.attach(self._walker, spawn_site)
		#self._walker.create_root_joints(arena.attach(self._walker, spawn_site))
		spawn_site.remove()
	

		# Add objects to arena.
		# Ball.
		ball_radius = 0.03
		ball = self._arena.mjcf_model.worldbody.add(
				'body', pos=(0.5, 0, ball_radius))
		ball.add('geom', type='sphere', size=(ball_radius,))
		self._ball_joint = ball.add('freejoint', name='ball')


		# Set timesteps and buffer.
		self.set_timesteps(physics_timestep=physics_timestep,
											 control_timestep=control_timestep)
		self._buffer_size = int(round(control_timestep/physics_timestep))

		# Explicitly enable observables.
		for obs in self._walker.observables.vestibular:
			obs.enabled = True
		self._walker.observables.root_position.enabled = True
		self._walker.observables.root_orientation.enabled = True
		self._walker.observables.egocentric_camera.enabled = False  # Maybe enable later.
	
		# Add custom ball position observable.
		self._walker.observables.add_observable('ball_rat_obs', self.ball_rat_obs)

	def initialize_episode_mjcf(self, random_state: np.random.RandomState):
		if hasattr(self._arena, 'regenerate'):
			self._arena.regenerate(random_state)
			# Optionally modify things in the arena before it's compiled to physics.
			# Nothing here for now...

	def initialize_episode(self, physics, random_state):
		super().initialize_episode(physics, random_state)
		
		self.episode_rewards = []
		self._should_terminate = False


		
		physics.data.qpos[:][:2] = [-0.5,0]
		physics.data.qvel[:][:3] = [0,0,0]


		# Set initial ball position and velocity.
		#pos = np.array((1., 1)) + random_state.uniform(-0.2, 0.2, size=2)
		#angle = np.deg2rad(random_state.uniform(30, 60))
		#vel = np.array([-np.cos(angle), -np.sin(angle)])  # Velocity direction.
		#vel *= random_state.uniform(0.5, 1.5)  # Speed.
		physics.named.data.qpos['ball'][:2] = [0.5,0.5]
		yvel = np.random.uniform(-4,-1)
		physics.named.data.qvel['ball'][:3] = [-2,yvel,0]
	
	def before_step(self, physics: 'mjcf.Physics', action,
				  random_state: np.random.RandomState):
		# Set ghost joint position and velocity.

		self._walker.apply_action(physics, action, random_state)

	def get_reward(self, physics: 'mjcf.Physics') -> float:
		"""Calculate reward."""
		


		ball_pos = physics.named.data.qpos['ball'][:2]
		#rat_pos = physics.data.qpos[:2]

		rat_pos = physics.named.data.geom_xpos['walker/head'][:2]

		distance = np.linalg.norm(ball_pos-rat_pos)  # Distance to ball.

		
		#rwrd = rewards.tolerance(distance,bounds=(0, 0),sigmoid='linear',margin=.5,value_at_margin=0.0)
		rwrd = -distance

		if rat_pos[0]>0.8 or rat_pos[0]<-0.8 or rat_pos[1]>0.8 or rat_pos[1]<-0.8:
			rwrd = -5
		
		if distance<0.05:
			rwrd = 5
		
		
		self.episode_rewards.append(rwrd)

		self.check_termination(physics)
		return rwrd

	def should_terminate_episode(self, physics: 'mjcf.Physics'):
		return self._should_terminate

	def check_termination(self, physics: 'mjcf.Physics') -> bool:
		"""Check some termination conditions."""

		if self.episode_rewards[-1] == 5 or self.episode_rewards[-1] == -5:
			self._should_terminate = True

		step = round(physics.time() / _CONTROL_TIMESTEP)
		if step>(self.max_steps-1):
			self._should_terminate = True
 
		return self._should_terminate



	def get_discount(self, physics: 'mjcf.Physics'):
		del physics  # unused by get_discount.
		if self._should_terminate:
			return 0.0
		return 1.0

	def name(self):
		return 'simple_rodent_task'

	@property
	def root_entity(self):
		return self._arena

	@composer.observable
	def ball_rat_obs(self):
		def get_ball_pos(physics: 'mjcf.Physics'):
			rat_pos = physics.named.data.geom_xpos['walker/head'][:3]
			ball_pos = physics.named.data.qpos['ball'][:3]
			distance1 = np.linalg.norm(ball_pos-rat_pos)  # Distance to ball.
			ball_vel = physics.named.data.qvel['ball'][:3]
			rat_vel = physics.named.data.qvel[:3]
			return np.concatenate((rat_pos,ball_pos,rat_vel,ball_vel,[distance1]),axis=0)
			
		return observable.Generic(get_ball_pos)

