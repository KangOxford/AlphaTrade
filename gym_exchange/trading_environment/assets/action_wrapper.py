# ========================== 04 ==========================
import numpy as np
from gym_exchange import Config
from gym_exchange.exchange.order_flow import OrderFlow
from gym_exchange.trading_environment.assets.action import PriceDelta
from gym_exchange.trading_environment.env_interface import SpaceParams
from gym_exchange.trading_environment.baselines.residual_policy import ResidualPolicy_Factory


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
        super().__init__(initial_number = Config.trade_id_generator)
        '''type(TradeIdGenerator) : function
        '''    
        
@singleton          
class OrderIdGenerator(IdGenerator):
    def __init__(self):
        super().__init__(initial_number = Config.order_id_generator)
        
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
        content_dict, revised_content_dict = self.get_content_dicts()
        order_flow = OrderFlow(**content_dict)
        auto_cancel = OrderFlow(**revised_content_dict) # TODO 
        return order_flow, auto_cancel
     
    def get_content_dicts(self):
        residual_action, residual_done = self.residual_policy.step()
        content_dict = {
            "Type" : 1, # submission of a new limit order
            "direction" : -1 if self.action[0] == 0 else 1,
            # "size" :  (self.action[1] - SpaceParams.Action.quantity_size_one_side) + residual_action,
            "size" :  (self.action[1] - SpaceParams.Action.quantity_size_one_side)//2 + residual_action, #change restricted within the half
            "price": self.price,
            "trade_id":self.trade_id,
            "order_id":self.order_id,
            "time":self.time,
        }
        revised_content_dict = {
            "Type" : 3, # total deletion of a limit order
            "direction" : content_dict['direction'], # keep the same direction
            "size"      : content_dict['size'],
            "price"     : content_dict['price'],
            "trade_id"  : content_dict['trade_id'],
            "order_id"  : content_dict['order_id'],
            "time"      : content_dict['time'],
        } 
        # try: #4
        #     assert content_dict['size'] >= 0, "The real quote size should be non-negative"
        # except: 
        #     breakpoint()
        #     print()#$
        assert content_dict['size'] >= 0, "The real quote size should be non-negative"
        return content_dict, revised_content_dict
    
    
    @property
    def price(self):
        return PriceDelta(self.price_list)('ask' if self.action[0]==0 else 'bid', self.action[2]) # side, price_delta 
    @property
    def trade_id(self):
        return self.trade_id_generator.step()
    @property
    def order_id(self):
        return self.order_id_generator.step()
    @property
    def time(self):
        return '30000.000000000' #TODO: implement; partly done
    '''revise it outside the class, (revised in the class Exchange)'''
    
    
    
if __name__ == '__main':
    pass