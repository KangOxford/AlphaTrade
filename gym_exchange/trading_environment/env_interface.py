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
        # --------------------- 01.01 ---------------------
        super().__init__()
        self.space_definition()
        self.vwap_estimator = Vwap()
        # --------------------- 01.02 ---------------------
        self.cur_state: Optional[State] = None
        # self._n_actions_taken: Optional[int] = None        
        self.seed()
    
    @abc.abstractmethod    
    def space_definition(self):
        pass
    # ========================== 02 ==========================
    def reset(self):
        """Reset episode and return initial observation."""
        self.cur_state = self.initial_state()
        assert self.cur_state in self.state_space, f"unexpected state {self.cur_state}"
        # self._n_actions_taken = 0
        return self.obs_from_state(self.cur_state)
    # ------------------------- 02.01 ------------------------
    def initial_state(self):
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
 