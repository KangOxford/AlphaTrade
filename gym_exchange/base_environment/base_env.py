# -*- coding: utf-8 -*-
from gym import Env
from gym import spaces
from gym_exchange.metric import Vwap
from gym_exchange.action import SimpleAction
from gym_exchange.action import BaseAction
from gym_exchange.base_environment import Configuration


class BaseEnv():
    """A stock trading environment based on OpenAI gym"""
    metadata = {'render.modes': ['human']}
    # -------------------------- 01 ----------------------------
    def __init__(self):
        super().__init__()
        self.space_definition()
        
    def space_definition(self):
        self.action_space = spaces.Box(
            0, Configuration.max_action,shape =(1,),
            dtype = np.int32)
        self.observation_space = \
            spaces.Box(
                low = np.array([Flag.min_price] * 10 +\
                            [Flag.num2liquidate]*1 +\
                            [Flag.max_episode_steps]*1 +\
                            [Flag.min_quantity]*10 + \
                            [Flag.min_num_left]*1 +\
                            [Flag.min_step_left]*1 
                            ).reshape((Flag.state_dim_1,Flag.state_dim_2)),
                high = np.array(
                            [Flag.max_price] * 10 +\
                            [Flag.num2liquidate]*1 +\
                            [Flag.max_episode_steps]*1 +\
                            [Flag.max_quantity]*10 + \
                            [Flag.max_num_left]*1 +\
                            [Flag.max_step_left]*1 
                            ).reshape((Flag.state_dim_1,Flag.state_dim_2)),
                shape = (Flag.state_dim_1,Flag.state_dim_2),
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
            pass
        # ····················· 03.03 ·····················
        @property
        def done(self):
            pass
        # ····················· 03.04 ·····················
        @property
        def info(self):
            pass
        