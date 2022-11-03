# from jinja2 import pass_eval_context
import numpy as np
from gym_exchange.trading_environment import Config
from gym_exchange.exchange.order_flow import OrderFlow
from gym_exchange.trading_environment.env_interface import SpaceParams
from gym_exchange.trading_environment.utils.residual_policy import ResidualPolicy_Factory

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
    def __call__(self, price_delta):
        tick_size = Config.tick_size
        # if the 
        return 0 # TODO: implement
        
# ========================== 03 ==========================

from enum import Enum
class Side(Enum):
    ask = 0
    bid = 1
            
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
        price_delta = self.price_delta + SpaceParams.price_delta_size_one_side
        side = 1 if self.side == 'bid' else 0
        # side = 0 if self.side == 'bid' else 1
        quantity  = self.quantity + SpaceParams.Action.quantity_size_one_side
        result = [side, quantity, price_delta]
        wrapped_result = np.array(result)
        return wrapped_result
        '''[side, quantity, price_delta]''' 

# ========================== 04 ==========================

def singleton(cls):
    _instance = {}
    def _singleton(*args, **kwargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kwargs)
        return _instance[cls]
    return _singleton


class IdGenerator():
    '''singleton method
    There should always be only one id generator object'''
    def __init__(self, initial_number):
        self.trade_id = initial_number
    def step(self):
        self.trade_id += 1
        return self.trade_id
    
@singleton    
class TradeIdGenerator(IdGenerator):
    def __init__(self):
        super().__init__(initial_number = 80000000)
        '''type(TradeIdGenerator) : function
        '''    
        
@singleton          
class OrderIdGenerator(IdGenerator):
    def __init__(self):
        super().__init__(initial_number = 88000000)
        
# ========================== 05 ==========================

class OrderFlowGenerator(object):
    def __init__(self):
        self.residual_policy = ResidualPolicy_Factory.produce("Twap")
        self.trade_id_generator = TradeIdGenerator() 
        self.order_id_generator = OrderIdGenerator()
    
        
    def step(self, action: np.ndarray, price_list) -> OrderFlow:
        # shoud the price list be one sided or two sided???? #TODO
        self.action = action # [side, quantity, price_delta]
        self.price_list = price_list
        content_dict = self.content_dict
        revised_content_dict = self.content_dict_revising(content_dict) # TODO
        order_flow = OrderFlow(**self.content_dict)
        auto_cancel = OrderFlow(**revised_content_dict) # TODO 
        return order_flow, auto_cancel
    
    @property    
    def content_dict(self):
        residual_action, residual_done = self.residual_policy.step()
        content_dict = {
            "Type" : 1, # submission of a new limit order
            "direction" : -1 if self.action[0] == 0 else 1,
            "size" : self.action[1] + residual_action,
            "price": self.price,
            "trade_id":self.trade_id,
            "order_id":self.order_id,
            "time":self.time,
        }
        return content_dict
    
    @property    
    def revised_content_dict(self, content_dict):
        new_content_dict = {
            "Type" : 3, # total deletion of a limit order
            "direction" : content_dict['direction'], # keep the same direction
            "size" : content_dict['size'],
            "price": content_dict['price'],
            "trade_id":content_dict['trade_id'],
            "order_id":content_dict['order_id'],
            "time":self.time,
        } 
        return new_content_dict
    
    @property
    def price(self):
        return PriceDelta(self.price_list)(self.action[2])
    @property
    def trade_id(self):
        return self.trade_id_generator.step()
    @property
    def order_id(self):
        return self.order_id_generator.step()
    @property
    def time(self):
        return str(30000.000000000) #TODO: implement; partly done
    '''revise it outside the class, (revised in the class Exchange)'''
    
    
    
if __name__ == '__main':
    pass