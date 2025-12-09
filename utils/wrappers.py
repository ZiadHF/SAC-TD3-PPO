import cv2
import gymnasium as gym
from gymnasium import Wrapper, ObservationWrapper
from collections import deque
import numpy as np

class PreprocessCarRacing(ObservationWrapper):
    def __init__(self, env, resize=(84, 84)):
        super().__init__(env)
        self.resize = resize
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(resize[0], resize[1], 1),
            dtype=np.float32
        )

    def observation(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        cropped = gray[12:, :]  # Remove score bar
        resized = cv2.resize(cropped, self.resize, interpolation=cv2.INTER_AREA)
        return resized[:, :, np.newaxis].astype(np.float32) / 255.0

class FrameStack(Wrapper):
    def __init__(self, env, num_stack=4):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)
        obs_shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(obs_shape[0], obs_shape[1], obs_shape[2] * num_stack),
            dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, term, trunc, info

    def _get_obs(self):
        return np.concatenate(self.frames, axis=-1)