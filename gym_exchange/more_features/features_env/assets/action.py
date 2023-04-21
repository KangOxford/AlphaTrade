import numpy as np
from gym_exchange import Config
from gym_exchange.environment.base_env.interface_env import SpaceParams
# ========================== 01 ==========================

class BaseAction():
    def __init__(self,side,quantity,price_delta):
        self.side = side
        self.quantity = quantity
        self.price_delta = price_delta
    
    @property
    def to_message(self):
        pass
    
    def __str__(self):
        fstring = f'side: {self.side}, quantity: {self.quantity}, price_delta: {self.price_delta}'
        return fstring


# -------------------------- 01 ----------------------------    
class SideAction(BaseAction):
    def __init__(self,side,quantity):
        super.__init__(side, quantity, price_delta = 0)
        ''''side = 'bid' or 'ask'
            quantity = 0 ~ num2liquidate (int)
            price_delta = 0 # fixed
            auto_cancel = 10 # fixed'''
    
# class DeltaAction(BaseAction):
#     def __init__(self,quantity,price_delta):
#         super.__init__(side, quantity, price_delta, side = 'ask')
#         ''''quantity = 0 ~ num2liquidate (int)
#             price_delta = -1, 0, 1 
#             side = 'ask' # fixed
#             auto_cancel = 10 # fixed'''
# # -------------------------- 02 ----------------------------    
# class SimpleAction(BaseAction):
#     def __init__(self,quantity):
#         super.__init__(side, quantity, price_delta = 0, side = 'ask')
#         ''''quantity = 0 ~ num2liquidate (int)
#             side =  'ask' # fixed
#             price_delta = 0 # fixed
#             auto_cancel = 10 # fixed'''

# ========================== 02 ==========================
class PriceDelta():
    def __init__(self, price_list):
        self.price_list = price_list
    def __call__(self, side, price_delta):
        self.side = side
        delta = self.adjust_price_delta(price_delta)
        if delta == 0: return self.price_list[0] # best_bid or best_ask # TODO: test 
        elif delta> 0: return self.positive_modifying(delta)
        else         : return self.negetive_modifying(delta)
    def adjust_price_delta(self, price_delta):
        return price_delta - SpaceParams.Action.price_delta_size_one_side
    def negetive_modifying(self, delta):
        tick_size = Config.tick_size
        if self.side == 'bid': return self.price_list[0] + delta * tick_size
        elif self.side=='ask': return self.price_list[0] - delta * tick_size
    def positive_modifying(self, delta):
        return self.price_list[delta] # delta_th best bid/ best ask

        
        
# ========================== 03 ==========================

# from enum import Enum
# class Side(Enum):
#     ask = 0
#     bid = 1
            
class Action(BaseAction):
    def __init__(self,side,quantity,price_delta):
        self.side = side 
        self.quantity = quantity
        self.price_delta = price_delta
        # ''''price_delta = -3,-2,-1,0,1,2,3 (0 ~ 7)
        #     side = 'bid'(0) or 'ask'(1)
        #     residual_quantity = -num2liquidate ~ num2liquidate (int: 0 ~ 2*num2liquidate+1)''' 
        
    @property
    def to_array(self) -> np.ndarray:
        '''wrapped_result: BaseAction'''
        price_delta = self.price_delta + SpaceParams.Action.price_delta_size_one_side
        side = 1 if self.side == 'bid' else 0
        # side = 0 if self.side == 'bid' else 1
        quantity  = self.quantity + SpaceParams.Action.quantity_size_one_side
        result = [side, quantity, price_delta]
        wrapped_result = np.array(result)
        return wrapped_result
        '''[side, quantity, price_delta]''' 

    
    
    
if __name__ == '__main':
    pass
