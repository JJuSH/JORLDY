import gym
from smac.env import StarCraft2Env
import numpy as np
from .base import BaseEnv

class Smac(BaseEnv):
    """Gym environment.

    Args:
        name (str): name of environment in Gym.
        render (bool): parameter that determine whether to render.
        custom_action (bool): parameter that determine whether to use custom action.
    """

    def __init__(
        self,
        difficulty,
        custom_action=False,
        **kwargs,
    ):
        self.env = StarCraft2Env(map_name = "8m", difficulty=difficulty)
        self.action_type = "discrete"

    def reset(self):
        self.score = 0
        state = self.env.reset()
        state = np.expand_dims(state, 0)  # for (1, state_size)
        return state

    def step(self, action):
        
        next_state, reward, done, info = self.env.step(action)
        self.score += reward

        next_state, reward, done = map(
            lambda x: np.expand_dims(x, 0), [next_state, [reward], [done]]
        )  # for (1, ?)
        return (next_state, reward, done)

    def close(self):
        self.env.close()


