# -*- coding: utf-8 -*-
import abc
import gym
import numpy as np
from gym import spaces
from typing import Generic, Optional, Sequence, Tuple, TypeVar
from gym_exchange.trading_environment import Config

State = TypeVar("State")
Observation = TypeVar("Observation")
Action = TypeVar("Action")

class SpaceParams(object):
    class Action:
        price_delta_size = 7
        side_size = 2
        quantity_size = 2*(Config.num2liquidate//Config.max_horizon +1) + 1
    class State:
        pass

class EnvInterface(gym.Env, abc.ABC, Generic[State, Observation, Action]):
    """A stock trading environment based on OpenAI gym"""
    metadata = {'render.modes': ['human']}
    # ========================== 01 ==========================
    def __init__(self):
        # --------------------- 01.01 ---------------------
        super().__init__()
        self.space_definition()
        self.vwap_estimator = Vwap() # Used for info
        # --------------------- 01.02 ---------------------
        self.cur_state: Optional[State] = None  
        self.seed()
    
    def space_definition(self):
        action_space = spaces.MultiDiscrete([SpaceParams.Action.price_delta_size, 
                                             SpaceParams.Action.side_size, 
                                             SpaceParams.Action.quantity_size]),
        # state_space=spaces.MultiDiscrete([price_delta_size, side_size, quantity_size]),
        # action_space=spaces.Discrete(5),

    
    # ========================== 02 ==========================
    def reset(self):
        """Reset episode and return initial observation."""
        self.cur_state = self.initial_state()
        assert self.cur_state in self.state_space, f"unexpected state {self.cur_state}"
        return self.obs_from_state(self.cur_state)
    # ------------------------- 02.01 ------------------------
    @abc.abstractmethod
    def initial_state(self) -> State:
        """Samples from the initial state distribution."""
        
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
 