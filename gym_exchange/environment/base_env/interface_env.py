# -*- coding: utf-8 -*-
import abc
import gym
import numpy as np
from gym import spaces
from typing import Generic, Optional, Sequence, Tuple, TypeVar
from gym_exchange import Config


State = TypeVar("State")
# Observation = TypeVar("Observation")
Action = TypeVar("Action")

class SpaceParams(object):
    class Action:
        side_size = 2

        # quantity_size_one_side = Config.num2liquidate//Config.max_horizon +1
        # quantity_size_one_side = 8
        quantity_size_one_side = Config.quantity_size_one_side
        # quantity_size_one_side = 3
        # quantity_size_one_side = 1
        quantity_size = 2*quantity_size_one_side + 1

        price_delta_size_one_side = 1
        price_delta_size = 2 * price_delta_size_one_side
        '''
        for Action(side = 0, price_delta = 0,quantity_delta = 0
        side = 0 means asks, the order is a selling order.
        price_delta = 0 means: submit a selling order at the asks side.
        price_dleta = 1 means: submit a selling order at the bids side. (across the spread)
        '''
    class State:
        low = np.array([Config.min_price] * 10 +\
                        [Config.min_quantity]*10 +\
                        [Config.min_price] * 10 +\
                        [Config.min_quantity]*10 
                        ).reshape((2 * Config.state_dim_1,Config.state_dim_2))
        high = np.array([Config.max_price] * 10 +\
                        [Config.max_quantity]*10 +\
                        [Config.max_price] * 10 +\
                        [Config.max_quantity]*10 
                        ).reshape((2 * Config.state_dim_1,Config.state_dim_2))
        shape = (2 * Config.state_dim_1,Config.state_dim_2)

# *************************** 1 *************************** #
class InterfaceEnv(gym.Env, abc.ABC, Generic[State, Action]):
# class EnvInterface(gym.Env, abc.ABC, Generic[State, Observation, Action]):
    """A stock trading environment based on OpenAI gym"""
    metadata = {'render.modes': ['human']}
    # ========================== 01 ==========================
    def __init__(self):
    # --------------------- 01.01 ---------------------
        super().__init__()
        self.action_space, self.state_space = self.space_definition()
        # print()#$
    # --------------------- 01.02 ---------------------
        self.cur_state: Optional[State] = None  
        self.seed()
    # --------------------- 01.03 ---------------------
    def space_definition(self):
        action_space = spaces.MultiDiscrete([SpaceParams.Action.side_size,
                                             SpaceParams.Action.quantity_size,
                                             SpaceParams.Action.price_delta_size])
        state_space = spaces.Box(
              low   = SpaceParams.State.low,
              high  = SpaceParams.State.high,
              shape = SpaceParams.State.shape,
              dtype = np.int64,
        )
        return action_space, state_space

    
    # ========================== 02 ==========================
    def reset(self):
        """Reset episode and return initial observation."""
        self.cur_state = self.initial_state()
        assert self.cur_state in self.state_space, f"unexpected state {self.cur_state}"
        # return self.obs_from_state(self.cur_state)
        return self.cur_state
    # ------------------------- 02.01 ------------------------
    @abc.abstractmethod
    def initial_state(self) -> State:
        """Samples from the initial state distribution."""
        
    # ========================== 03 ==========================
    @abc.abstractmethod
    def step(self, action):
        '''input : action
           return: state, reward, done, info'''
        return self.state, self.reward, self.done, self.info
        #    return: observation, reward, done, info'''
        # return self.observation, self.reward, self.done, self.info
    # --------------------- 03.01 ---------------------
    @property
    def state(self):
        pass
    # @property
    # def observation(self):
    #     pass
    # # ···················· 03.01.01 ···················· 
    # @abc.abstractmethod
    # def obs_from_state(self, state: State) -> Observation:
    #     """Sample observation for given state."""
    # --------------------- 03.02 ---------------------
    @property
    def reward(self):
        pass
    # --------------------- 03.03  ---------------------
    @property
    def done(self):
        pass
    # --------------------- 03.04 ---------------------  
    @property
    def info(self):
        pass
    # ========================== 04 ==========================
    @abc.abstractmethod
    def render(self, mode = 'human'):
        '''for render method'''

if __name__ == "__main__":
    pass
