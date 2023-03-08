import numpy as np
from gym_exchange import Config
from gym_exchange.trading_environment.basic_env.interface_env import SpaceParams
# ========================== 01 ==========================

class BaseAction():
    def __init__(self,direction,quantity,price_delta):
        self.direction = direction
        self.quantity = quantity
        self.price_delta = price_delta

    @property
    def to_message(self):
        pass

# ========================== 02 ==========================

# from enum import Enum
# class Side(Enum):
#     ask = 0
#     bid = 1
            
class Action(BaseAction): # more precise class name: DeltaAction
    def __init__(self,direction,quantity_delta,price_delta):
        self.direction = direction
        self.quantity_delta = quantity_delta
        self.price_delta = price_delta
        # ''''price_delta = -3,-2,-1,0,1,2,3 (0 ~ 7)
        #     side = 'bid'(0) or 'ask'(1)
        #     residual_quantity = -num2liquidate ~ num2liquidate (int: 0 ~ 2*num2liquidate+1)'''
        
    @property # to_machine_readable
    def encoded(self) -> np.ndarray:
        '''wrapped_result: BaseAction'''
        price_delta = self.price_delta + SpaceParams.Action.price_delta_size_one_side
        side = 1 if self.direction == 'bid' else 0
        # side = 0 if self.side == 'bid' else 1
        quantity_delta = self.quantity_delta + SpaceParams.Action.quantity_size_one_side
        result = [side, quantity_delta, price_delta]
        wrapped_result = np.array(result)
        return wrapped_result
        '''[side, quantity_delta, price_delta]'''

    @classmethod
    def decode(cls,action):  # to_human_readable
        direction = -1 if action[0] == 0 else 1
        quantity_delta = action[1] - SpaceParams.Action.quantity_size_one_side
        price_delta = action[2] - SpaceParams.Action.price_delta_size_one_side
        return np.array([direction,quantity_delta,price_delta])

    def __str__(self):
        fstring = f'side: {self.direction}, quantity_delta: {self.quantity_delta}, price_delta: {self.price_delta}'
        return fstring
    
    
    
if __name__ == '__main':
    pass
