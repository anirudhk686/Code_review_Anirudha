import gymnasium as gym
import numpy as np
import collections

# Custom Observation Wrapper for Corridor Task
class CustomObservationWrapperCorridor(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(140+9+32,), dtype=np.float32)

    def observation(self, ob):
        FLAT_OBSERVATION_KEY = 'observations'
        
        def flatten_observation(observation, output_key=FLAT_OBSERVATION_KEY):
            if not isinstance(observation, collections.abc.MutableMapping):
                raise ValueError('Can only flatten dict-like observations.')

            keys = sorted(observation.keys())
            keys.remove('walker/vel_obs')
            keys.remove('walker/prev_action')
            
            observation_arrays = [observation[key].ravel() for key in keys]
            return type(observation)([(output_key, np.concatenate(observation_arrays))])

        vel_obs = ob['walker/vel_obs']
        prev_action = ob['walker/prev_action']

        ob = flatten_observation(ob)[FLAT_OBSERVATION_KEY]
        ob = np.concatenate((vel_obs, ob, prev_action))
    
        return ob

# Custom Observation Wrapper for Path Task
class CustomObservationWrapperPath(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(140+14+32,), dtype=np.float32)

    def observation(self, ob):
        FLAT_OBSERVATION_KEY = 'observations'
        
        def flatten_observation(observation, output_key=FLAT_OBSERVATION_KEY):
            if not isinstance(observation, collections.abc.MutableMapping):
                raise ValueError('Can only flatten dict-like observations.')

            keys = sorted(observation.keys())
            keys.remove('walker/future_xpos')
            keys.remove('walker/prev_action')
            
            observation_arrays = [observation[key].ravel() for key in keys]
            return type(observation)([(output_key, np.concatenate(observation_arrays))])

        future_xpos = ob['walker/future_xpos']
        prev_action = ob['walker/prev_action']

        ob = flatten_observation(ob)[FLAT_OBSERVATION_KEY]
        ob = np.concatenate((future_xpos, ob, prev_action))
    
        return ob

# Custom Observation Wrapper for Fetch Task
class CustomObservationWrapperFetch(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(153+32,), dtype=np.float32)

    def observation(self, ob):
        FLAT_OBSERVATION_KEY = 'observations'
        
        def flatten_observation(observation, output_key=FLAT_OBSERVATION_KEY):
            if not isinstance(observation, collections.abc.MutableMapping):
                raise ValueError('Can only flatten dict-like observations.')

            keys = sorted(observation.keys())
            keys.remove('walker/ball_rat_obs')
            keys.remove('walker/prev_action')
            
            observation_arrays = [observation[key].ravel() for key in keys]
            return type(observation)([(output_key, np.concatenate(observation_arrays))])

        ball_rat_pos = ob['walker/ball_rat_obs']
        prev_action = ob['walker/prev_action']
        ob = flatten_observation(ob)[FLAT_OBSERVATION_KEY]
        ob = np.concatenate((ball_rat_pos, ob, prev_action))
    
        return ob

# Generalized function to select the appropriate observation wrapper based on environment type
def select_custom_observation_wrapper(env, task_type):
    if task_type == "corridor":
        return CustomObservationWrapperCorridor(env)
    elif task_type == "path":
        return CustomObservationWrapperPath(env)
    elif task_type == "fetch":
        return CustomObservationWrapperFetch(env)
    else:
        raise ValueError(f"Unknown task type: {task_type}")
