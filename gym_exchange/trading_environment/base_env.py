# -*- coding: utf-8 -*-
import numpy as np
from gym import Env
from gym import spaces
from gym_exchange.metric import Vwap
from gym_exchange.action import SimpleAction
from gym_exchange.action import BaseAction
from gym_exchange.base_environment import Conf # Configuration


class BaseEnv():
    """A stock trading environment based on OpenAI gym"""
    metadata = {'render.modes': ['human']}
    # -------------------------- 01 ----------------------------
    def __init__(self):
        super().__init__()
        self.space_definition()
        self.vwap_estimator = Vwap()
        
    def space_definition(self):
        self.action_space = spaces.Box(
            0, Conf.max_action,shape =(1,),
            dtype = np.int32)
        self.observation_space = \
            spaces.Box(
                low = np.array([Conf.min_price] * 10 +\
                            [Conf.num2liquidate]*1 +\
                            [Conf.max_episode_steps]*1 +\
                            [Conf.min_quantity]*10 + \
                            [Conf.min_num_left]*1 +\
                            [Conf.min_step_left]*1 
                            ).reshape((Conf.state_dim_1,Conf.state_dim_2)),
                high = np.array(
                            [Conf.max_price] * 10 +\
                            [Conf.num2liquidate]*1 +\
                            [Conf.max_episode_steps]*1 +\
                            [Conf.max_quantity]*10 + \
                            [Conf.max_num_left]*1 +\
                            [Conf.max_step_left]*1 
                            ).reshape((Conf.state_dim_1,Conf.state_dim_2)),
                shape = (Conf.state_dim_1,Conf.state_dim_2),
                dtype = np.int32,)
    # -------------------------- 02 ----------------------------
    def reset(self):
        pass
    
    # -------------------------- 03 ----------------------------
    def step(self, action):
        '''input : action
           return: observation, reward, done, info'''
        return observation, reward, done, info
        # ····················· 03.01 ·····················
        @property
        def observation(self):
            pass
        # ····················· 03.02 ·····················
        @property
        def reward(self):
            difference = self.vwap_estimator.difference
            return reward
        # ····················· 03.03 ·····················
        @property
        def done(self):
            pass
        # ····················· 03.04 ·····················
        @property
        def info(self):
            pass
        