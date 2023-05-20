from typing import Dict, Tuple
import gym
import gym.spaces
import numpy as np

class Goal(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
    ):
        super().__init__(env)

        self.goal_state = None

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        state, reward, done, info = self.env.step(action)

        if self.goal_state is None and done:
            self.goal_state = state

        return state, reward, done, info

    def is_terminal(self, state: np.ndarray):
        return np.array_equal(state, self.goal_state)
