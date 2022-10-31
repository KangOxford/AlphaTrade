# -*- coding: utf-8 -*-
import abc
import numpy as np
from gym import Env
from gym import spaces
from gym_exchange.metric import Vwap
from gym_exchange.action import SimpleAction
from gym_exchange.action import BaseAction
from gym_exchange.base_environment import Conf # Configuration


class EnvInterface(abc.ABC):
    """A stock trading environment based on OpenAI gym"""
    metadata = {'render.modes': ['human']}
    # -------------------------- 01 ----------------------------
    def __init__(self):
        super().__init__()
        self.space_definition()
        self.vwap_estimator = Vwap()
    
    @abc.abstractmethod    
    def space_definition(self):
        pass
    # -------------------------- 02 ----------------------------
    @abc.abstractmethod
    def reset(self):
        pass
    
    # -------------------------- 03 ----------------------------
    @abc.abstractmethod
    def step(self, action):
        '''input : action
           return: observation, reward, done, info'''
        return self.observation, self.reward, self.done, self.info
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
        