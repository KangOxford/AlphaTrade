from abc import ABC, abstractmethod
from gym import Env


class BaseEnvironment(Env, ABC):
    def __init__(self) -> None:
        super().__init__()
    def setp(self, action: float = 0):
        pass
    def reset(self):
        '''return the observation of the initial condition'''
        return super().reset()
    def seed(self):
        pass
    def close(self) -> None:
        return super().close()
    def render(self):
        pass