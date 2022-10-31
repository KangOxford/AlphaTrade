# -*- coding: utf-8 -*-
import abc
import gym
import numpy as np
from gym import spaces
from typing import Generic, Optional, Sequence, Tuple, TypeVar

State = TypeVar("State")
Observation = TypeVar("Observation")
Action = TypeVar("Action")

class EnvInterface(gym.Env, abc.ABC, Generic[State, Observation, Action]):
    """A stock trading environment based on OpenAI gym"""
    metadata = {'render.modes': ['human']}
    # ========================== 01 ==========================
    def __init__(self):
        super().__init__()
        self.space_definition()
        self.vwap_estimator = Vwap()
        self.seed()
    
    @abc.abstractmethod    
    def space_definition(self):
        pass
    # ========================== 02 ==========================
    @abc.abstractmethod
    def reset(self):
        pass
    
    # ========================== 03 ==========================
    @abc.abstractmethod
    def step(self, action):
        '''input : action
           return: observation, reward, done, info'''
        return self.observation, self.reward, self.done, self.info
        # --------------------- 03.01 ---------------------
        @property
        def observation(self):
            pass
        # ···················· 03.01.01 ···················· 
        @abc.abstractmethod
        def obs_from_state(self, state: State) -> Observation:
            """Sample observation for given state."""
        # --------------------- 03.02 ---------------------
        @property
        def reward(self):
            difference = self.vwap_estimator.difference
            return reward
        # --------------------- 03.03  ---------------------
        @property
        def done(self):
            pass
        # --------------------- 03.04 ---------------------  
        @property
        def info(self):
            pass
 