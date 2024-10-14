"""A simple rodent task example."""

from typing import Dict,Callable, Union, Sequence

import numpy as np
import pickle
from dm_control import mujoco


from dm_control.mujoco.wrapper import mjbindings
mjlib = mjbindings.mjlib

from dm_control import composer
from dm_control.composer.observation import observable
import rodent
from dm_control.locomotion.arenas import floors
import quaternions
from dm_control import mjcf



_CONTROL_TIMESTEP = 0.0055
_PHYSICS_TIMESTEP = .000275

_SPAWN_POS = [0,  0.0,  0.0]
SITE_NAMES = [
"walker/nose_0_kpsite",
"walker/ear_L_1_kpsite",
"walker/ear_R_2_kpsite",
"walker/neck_3_kpsite",
"walker/spineL_4_kpsite",
"walker/tailbase_5_kpsite",
"walker/shoulder_L_6_kpsite",
"walker/elbow_L_7_kpsite",
"walker/wrist_L_8_kpsite",
"walker/hand_L_9_kpsite",
"walker/shoulder_R_10_kpsite",
"walker/elbow_R_11_kpsite",
"walker/wrist_R_12_kpsite",
"walker/hand_R_13_kpsite",
"walker/knee_L_14_kpsite",
"walker/ankle_L_15_kpsite",
"walker/foot_L_16_kpsite",
"walker/knee_R_17_kpsite",
"walker/ankle_R_18_kpsite",
"walker/foot_R_19_kpsite"
]


def rodent_env(random_state=None,floor_size = (0.762, 0.762),
			   std = {'com': 900, 'qvel': 0.0008, 'root2site': 700, 'joint_quat': 0.35,'site_xpos':0},
			   weights = (0.2,0.2,0.2,0.2,0.2),physics_timestep =  _PHYSICS_TIMESTEP,rank = 0,add_ball=False,
				site_weights = None, dataset_dir=None,datasetname=None, max_steps = 1000,reward_const = 0,is_test = False,
				add_ghost=True, clip_ids_to_use = [0], initial_repeat = 10,add_ramp = False,add_fixed_cams = True):
		"""Create simple rodent environment."""
		global _SPAWN_POS
		global _PHYSICS_TIMESTEP
		_PHYSICS_TIMESTEP = physics_timestep


		try:
			with open(dataset_dir+datasetname, 'rb') as f:
				all_clip_training_data = pickle.load(f)
		except:
			all_clip_training_data = None


		walker = rodent
		arena = floors.Floor(size=floor_size)
		task = RodentTask(walker=walker,
											arena=arena,
											physics_timestep=_PHYSICS_TIMESTEP,
											control_timestep=_CONTROL_TIMESTEP,
											all_clip_training_data = all_clip_training_data,
											site_names = SITE_NAMES,
											std = std,
											weights = weights,
											max_steps = max_steps,
											rank = rank,
											reward_const = reward_const,
											add_ghost = add_ghost,
											site_weights = site_weights,
											is_test = is_test,
											clip_ids_to_use = clip_ids_to_use,
											dataset_dir = dataset_dir,
											add_ball = add_ball,
											initial_repeat = initial_repeat,
											add_ramp = add_ramp,
											add_fixed_cams = add_fixed_cams
											)
		return composer.Environment(time_limit= (max_steps/180) ,
																task=task,
																random_state=random_state,
																strip_singleton_obs_buffer_dim=True)

class Ramp(composer.Entity):

	def __init__(self, *args, **kwargs):
		self._mjcf_root = None  # Declare that _mjcf_root exists to allay pytype.
		super().__init__(*args, **kwargs)

	def _build(self, *args, **kwargs) -> None:
		"""Initializes this arena.

		The function takes two arguments through args, kwargs:
		name: A string, the name of this arena. If `None`, use the model name
			defined in the MJCF file.
		xml_path: An optional path to an XML file that will override the default
			composer arena MJCF.

		Args:
		*args: See above.
		**kwargs: See above.
		"""
		if args:
			name = args[0]
		else:
			name = kwargs.get('name', None)
		if len(args) > 1:
			xml_path = args[1]
		else:
			xml_path = kwargs.get('xml_path', None)

		self._mjcf_root = mjcf.from_path(xml_path)
		if name:
			self._mjcf_root.model = name

	def add_free_entity(self, entity):
		"""Includes an entity in the arena as a free-moving body."""
		frame = self.attach(entity)
		frame.add('freejoint')
		return frame

	@property
	def mjcf_model(self):
		return self._mjcf_root


class FixedCams(composer.Entity):

  def __init__(self, *args, **kwargs):
    self._mjcf_root = None  # Declare that _mjcf_root exists to allay pytype.
    super().__init__(*args, **kwargs)

  def _build(self, *args, **kwargs) -> None:
    """Initializes this arena.

    The function takes two arguments through args, kwargs:
      name: A string, the name of this arena. If `None`, use the model name
        defined in the MJCF file.
      xml_path: An optional path to an XML file that will override the default
        composer arena MJCF.

    Args:
      *args: See above.
      **kwargs: See above.
    """
    if args:
      name = args[0]
    else:
      name = kwargs.get('name', None)
    if len(args) > 1:
      xml_path = args[1]
    else:
      xml_path = kwargs.get('xml_path', None)

    self._mjcf_root = mjcf.from_path(xml_path)
    if name:
      self._mjcf_root.model = name

  def add_free_entity(self, entity):
    """Includes an entity in the arena as a free-moving body."""
    frame = self.attach(entity)
    frame.add('freejoint')
    return frame

  @property
  def mjcf_model(self):
    return self._mjcf_root



class RodentTask(composer.Task):
	"""Simple rodent task."""

	def __init__(
			self,
			walker: Union['base.Walker', Callable], # type: ignore
			arena: composer.Arena,
			physics_timestep: float,
			control_timestep: float,
			all_clip_training_data,
			site_names,
			std,
			weights,
			max_steps,
			rank,
			reward_const,
			add_ghost,
			site_weights,
			is_test,
			clip_ids_to_use,
			dataset_dir,
			add_ball,
			initial_repeat,
			add_ramp,
			add_fixed_cams
	):

		self._arena = arena
		self._walker = walker.Rat()
		self._should_terminate = False
		self.all_clip_training_data = all_clip_training_data
		self.site_names = site_names
		self.std = std
		self.weights  = weights
		self.max_steps = max_steps
		self.rank = rank
		self.episode_rewards = []
		self.reward_const = reward_const
		self.diff_data = []
		self.add_ghost = add_ghost
		self.site_weights = site_weights
		self.is_test = is_test
		self.add_ball = add_ball
		self.initial_repeat = initial_repeat
		self.add_ramp = add_ramp
		self.ctrl_ranges = None
		self.add_fixed_cams = add_fixed_cams

		self.clip_ids_to_use = clip_ids_to_use
		self.train_clip_number = 0 # assigned in reset env
		self.all_clip_lengths = None # npy file read below
		self.training_data = None # clipped from all_training_data assigned in reset env
		self.clip_len = 1000 # assigned in reset env

		self.prev_action = np.zeros(32)



		if self.all_clip_training_data is not None:

			self.all_clip_lengths =   np.load(dataset_dir+"all_lengths_final.npy")

		try:
			self.init_qpos = np.load(dataset_dir+"init_qpos.npy")
		except:
			self.init_qpos = None


		# Add walker to arena.
		spawn_site = self._arena.mjcf_model.worldbody.add('site', pos=_SPAWN_POS)
		self._walker.create_root_joints(arena.attach(self._walker, spawn_site))
		spawn_site.remove()


		if self.add_ghost:
			ghost_alpha = 0
			self._ghost = walker.Rat()
			make_ghost(self._ghost,ghost_alpha)
			spawn_site = arena.mjcf_model.worldbody.add('site', pos=_SPAWN_POS)
			self._ghost_frame = arena.attach(self._ghost, spawn_site)
			spawn_site.remove()
			self._ghost_joint = self._ghost_frame.add(
				'joint', type='free', armature=1)
		else:
			self.add_ghost = None


		geoms = self._walker.mjcf_model.find_all('geom')
		for geom in geoms:
			if 'vertebra' in geom.name:
				geom.rgba = [0, 0, 0, 0]  # Make the geom invisible.


		# Ball.
		if self.add_ball:
			ball_radius = 0.023
			ball = self._arena.mjcf_model.worldbody.add(
				'body', pos=(0.2, 0.3, ball_radius))
			ball.add('geom', type='sphere', size=(ball_radius,))
			for geom in ball.find_all('geom'):
				geom.set_attributes(rgba=(0.875, 1.0, 0.309,1.0))
			self._ball_joint = ball.add('freejoint', name='ball')

		if self.add_ramp:
			self._ramp = Ramp("ramp", "../wedge/wedge_visual.xml")
			self._arena.attach(self._ramp)

		if self.add_fixed_cams:
			self._fixedcams = FixedCams("Cam13", "../dataset/Cam13.xml")
			self._arena.attach(self._fixedcams)
			self._fixedcams = FixedCams("Cam14", "../dataset/Cam14.xml")
			self._arena.attach(self._fixedcams)


		# Set timesteps and buffer.
		self.set_timesteps(physics_timestep=physics_timestep,
											 control_timestep=control_timestep)
		self._buffer_size = int(round(control_timestep/physics_timestep))

		# Explicitly enable observables.
		enabled_observables = []
		enabled_observables += self._walker.observables.proprioception
		enabled_observables += self._walker.observables.kinematic_sensors
		enabled_observables += self._walker.observables.dynamic_sensors
		enabled_observables.append(self._walker.observables.sensors_touch)
		# enabled_observables.append(self._walker.observables.egocentric_camera)
		for observable in enabled_observables:
			observable.enabled = True
		# Add custom ball position observable.
		#self._walker.observables.add_observable('ball_position',self.ball_position)
		self._walker.observables.add_observable('current_step',self.current_step)
		self._walker.observables.add_observable('future_xpos',self.future_xpos)
		self._walker.observables.add_observable('prev_action',self.previous_action)


	def initialize_episode_mjcf(self, random_state: np.random.RandomState):
		if hasattr(self._arena, 'regenerate'):
			self._arena.regenerate(random_state)
			# Optionally modify things in the arena before it's compiled to physics.
			# Nothing here for now...

	def initialize_episode(self, physics, random_state):
		super().initialize_episode(physics, random_state)

		self.episode_rewards = []
		self._should_terminate = False

		if self.ctrl_ranges is None:
			self.ctrl_ranges = physics.model.actuator_ctrlrange.copy()


		if self.all_clip_training_data is not None:
			if self.training_data is None:
				self.training_data = {}
				self.train_clip_number	= self.clip_ids_to_use[self.rank%len(self.clip_ids_to_use)]

				clip_start = self.all_clip_lengths[self.train_clip_number][2]
				clip_end = self.all_clip_lengths[self.train_clip_number][1]

				for k in self.all_clip_training_data:
					self.training_data[k] = self.all_clip_training_data[k][clip_start:clip_end]
					t1 = np.repeat(np.expand_dims(self.training_data[k][0],0),self.initial_repeat,0)
					t2 = np.repeat(np.expand_dims(self.training_data[k][-1],0),70,0)
					self.training_data[k] = np.concatenate((t1,self.training_data[k],t2),axis=0)

				self.clip_len = self.training_data['qpos'].shape[0]

			if self.training_data is not None:

				if self.add_ball:
					physics.named.data.qpos['ball'][:2] = [0,0]
					physics.named.data.qvel['ball'][:2] = [0,0]

					if self.add_ghost:
						physics.data.qpos[:-7][:int(physics.data.qpos[:-7].shape[0]/2)] = self.init_qpos
						physics.data.qpos[:-7][:2] = self.training_data['qpos'][0][:2]
						physics.data.qpos[:-7][6] = self.training_data['qpos'][0][6]

						physics.data.qpos[:-7][int(physics.data.qpos[:-7].shape[0]/2):] = self.training_data['qpos'][0]
					else:

						physics.data.qpos[:-7][:int(physics.data.qpos[:-7].shape[0])] = self.init_qpos
						physics.data.qpos[:-7][:2] = self.training_data['qpos'][0][:2]
						physics.data.qpos[:-7][6] = self.training_data['qpos'][0][6]

				else:
					if self.add_ghost:
						physics.data.qpos[:int(physics.data.qpos.shape[0]/2)] = self.init_qpos
						physics.data.qpos[:2] = self.training_data['qpos'][0][:2]
						physics.data.qpos[6] = self.training_data['qpos'][0][6]

						physics.data.qpos[int(physics.data.qpos.shape[0]/2):] = self.training_data['qpos'][0]
					else:

						physics.data.qpos[:int(physics.data.qpos.shape[0])] = self.init_qpos
						physics.data.qpos[:2] = self.training_data['qpos'][0][:2]
						physics.data.qpos[6] = self.training_data['qpos'][0][6]




	def before_step(self, physics: 'mjcf.Physics', action,
				  random_state: np.random.RandomState):
		# Set ghost joint position and velocity.

		if self.add_ghost:
			step = int(np.round(physics.data.time / self.control_timestep))
			physics.data.qpos[int(physics.data.qpos.shape[0]/2):] = self.training_data['qpos'][step].copy()

		action = action * (self.ctrl_ranges[:,1] - self.ctrl_ranges[:,0])/2 + (self.ctrl_ranges[:,1] + self.ctrl_ranges[:,0])/2
		self.prev_action = action
		self._walker.apply_action(physics, action, random_state)


	def get_imitation_reward(self,physics,step,std,weights):

		def compute_diffs(model_features: Dict[str, np.ndarray],
									reference_features: Dict[str, np.ndarray],
									n: int = 2,
								 ) -> Dict[str, float]:
			"""Computes sums of absolute values of differences between components of
				model and reference features.

				Args:
						model_features, reference_features: Dictionaries of features to compute
								differences of.
						n: Exponent for differences. E.g., for squared differences use n = 2.

				Returns:
						Dictionary of differences, one value for each entry of input dictionary.
			"""
			diffs = {}
			for k in model_features:
					if 'quat' not in k:
							# Regular vector differences.
							diffs[k] = np.sum(
									np.abs(model_features[k] - reference_features[k])**n)
					else:
							# Quaternion differences (always positive, no need to use np.abs).
							diffs[k] = np.sum(
									quaternions.quat_dist_short_arc(
											model_features[k], reference_features[k])**n)
			return diffs


		# === Collect reference pose features.
		reference_data  = self.training_data
		qpos_ref = reference_data['qpos'][step]
		qvel_ref = reference_data['qvel'][step]
		root2site_ref = reference_data['root2site'][step]
		joint_quat_ref = reference_data['joint_quat'][step]
		joint_quat_ref = np.vstack((qpos_ref[3:7], joint_quat_ref))
		site_xpos_ref = reference_data['site_xpos'][step]*self.site_weights

		reference_features = {
				'com': reference_data['qpos'][step, :3],
				'qvel': qvel_ref,
				'root2site': root2site_ref,
				'joint_quat': joint_quat_ref,
				'site_xpos': site_xpos_ref
		}

		# === Collect model pose features.
		if self.add_ghost:
			qpos = physics.data.qpos[:int(physics.data.qpos.shape[0]/2)]
			qvel = physics.data.qvel[:int(physics.data.qvel.shape[0]/2)]
			xaxis1 = physics.data.xaxis[:int(physics.data.xaxis.shape[0]/2)][1:, :]  # (n_joints-1, 3)
		else:
			qpos = physics.data.qpos[:int(physics.data.qpos.shape[0])]
			qvel = physics.data.qvel[:int(physics.data.qvel.shape[0])]
			xaxis1 = physics.data.xaxis[:int(physics.data.xaxis.shape[0])][1:, :]  # (n_joints-1, 3)



		root_xpos = qpos[:3]
		root_quat = qpos[3:7]
		site_xpos = physics.named.data.site_xpos[self.site_names]*self.site_weights  # (n_sites, 3)
		root2site = quaternions.get_egocentric_vec(
				root_xpos,site_xpos,root_quat)  # (n_sites, 3)


		xaxis1 = quaternions.rotate_vec_with_quat(xaxis1, quaternions.reciprocal_quat(root_quat))
		qpos7 = qpos[7:]  # (n_joints-1,)
		joint_quat = quaternions.joint_orientation_quat(
										xaxis1,qpos7)  # (n_joints-1, 4)
		joint_quat = np.vstack((root_quat, joint_quat))  # (n_joints, 4)



		model_features = {
				'com': qpos[:3],
				'qvel': qvel,
				'root2site': root2site,  # (n_sites, 3)
				'joint_quat': joint_quat,  # (n_joints, 4)
				'site_xpos': site_xpos
		}

		diffs = compute_diffs(model_features, reference_features, n=2)

		reward_factors = []
		for k in model_features.keys():
				reward_factors.append(
						np.exp(-std[k] * diffs[k]))
		reward_factors = np.array(reward_factors)


		reward_factors = reward_factors*weights
		reward = reward_factors.sum()


		return reward

	def get_reward(self, physics: 'mjcf.Physics') -> float:
		"""Calculate reward."""

		step = round(physics.time() / _CONTROL_TIMESTEP)



		rwrd = self.get_imitation_reward(physics,step,self.std,self.weights)

		if not self.is_test:
			if len(self.episode_rewards)>25:
				if sum(self.episode_rewards[-25:])<0.1:
					rwrd = -0.5

		self.episode_rewards.append(rwrd)

		rwrd = rwrd + self.reward_const

		'''
		if self.is_test:
			#print(step)
			#scene_option = mujoco.wrapper.core.MjvOption()
			#scene_option.skingroup[0] = 0
			#img = PIL.Image.fromarray(physics.render(camera_id=3,height=2200,width=3208,scene_option=scene_option))
			#img.save('test_videos/'+str(step)+'.png')
		'''

		self.check_termination(physics)

		return rwrd

	def should_terminate_episode(self, physics: 'mjcf.Physics'):
		return self._should_terminate

	def check_termination(self, physics: 'mjcf.Physics') -> bool:
		"""Check some termination conditions."""

		if self.episode_rewards[-1]==-0.5:
			self._should_terminate = True

		step = round(physics.time() / _CONTROL_TIMESTEP)
		if step>(self.clip_len-65):
			self._should_terminate = True


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
	def current_step(self):
		def get_current_step(physics: 'mjcf.Physics'):
			step = round(physics.time() / _CONTROL_TIMESTEP)
			return step
		return observable.Generic(get_current_step)

	@composer.observable
	def future_xpos(self):
		def get_future_xpos(physics: 'mjcf.Physics'):
			step = round(physics.time() / _CONTROL_TIMESTEP)
			if self.training_data is not None:
				reference_data  = self.training_data['site_xpos']
				site_xpos_ref = reference_data[[step+20,step+40,step+60],:,:]
				root = physics.data.qpos[:7]
				site_xpos_ref = quaternions.get_egocentric_vec(root[:3],site_xpos_ref,root[3:])
				site_xpos_ref = site_xpos_ref.flatten()
			else:
				site_xpos_ref = np.zeros((180))
			return site_xpos_ref
		return observable.Generic(get_future_xpos)
	
	@composer.observable
	def previous_action(self):
		def get_prev_action(physics: 'mjcf.Physics'):
			return self.prev_action
		return observable.Generic(get_prev_action)
	




def make_ghost(walker,ghost_alpha):
	"""Create a 'ghost' fly to serve as a tracking target."""
  	# Remove model elements.
	for tendon in walker.mjcf_model.find_all('tendon'):
		tendon.remove()

	#for joint in walker.mjcf_model.find_all('joint'):
	#  joint.remove()

	for act in walker.mjcf_model.find_all('actuator'):
		act.remove()

	for sensor in walker.mjcf_model.find_all('sensor'):
		if sensor.tag == 'touch' or sensor.tag == 'force':
			sensor.remove()

	for exclude in walker.mjcf_model.find_all('contact'):
		exclude.remove()

	for light in walker.mjcf_model.find_all('light'):
		light.remove()
	for body in walker.mjcf_model.find_all('body'):
		body.gravcomp = 1

	for camera in walker.mjcf_model.find_all('camera'):
		camera.remove()
	for site in walker.mjcf_model.find_all('site'):
		site.rgba = (0, 0, 0, 0)

	for geom in walker.mjcf_model.find_all('geom'):

		rgba = (0.5,0.5,0.5,ghost_alpha)
		geom.set_attributes(
		user=(0,),
		contype=0,
		conaffinity=0,
		rgba=rgba)






