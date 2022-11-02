from jinja2 import pass_eval_context
import numpy as np
from gym_exchange.exchange.order_flow import OrderFlow
from gym_exchange.trading_environment.env_interface import SpaceParams

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
    
class DeltaAction(BaseAction):
    def __init__(self,quantity,price_delta):
        super.__init__(side, quantity, price_delta, side = 'ask')
        ''''quantity = 0 ~ num2liquidate (int)
            price_delta = -1, 0, 1 
            side = 'ask' # fixed
            auto_cancel = 10 # fixed'''
# -------------------------- 02 ----------------------------    
class SimpleAction(BaseAction):
    def __init__(self,quantity):
        super.__init__(side, quantity, price_delta = 0, side = 'ask')
        ''''quantity = 0 ~ num2liquidate (int)
            side =  'ask' # fixed
            price_delta = 0 # fixed
            auto_cancel = 10 # fixed'''

class PriceDelta():
    def __init__(self, price_list):
        self.price_list = price_list
    def __call__(self, price_delta):
        return 0 # TODO: implement
        
# ========================== 02 ==========================
        
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

# ========================== 03 ==========================

class OrderFlowGenerator(object):
    def __init__(self, residual_policy):
        self.residual_policy = residual_policy
        self.trade_id_generator = TradeIdGenerator() 
        self.order_id_generator = OrderIdGenerator()
    
        
    def step(self, action: np.ndarray, price_list) -> OrderFlow:
        self.action = action # [side, quantity, price_delta]
        self.price_list = price_list
        content_dict = self.content_dict
        order_flow = OrderFlow(**content_dict)
        auto_cancel = OrderFlow(**(
            content_dict
        )) # TODO 
        return order_flow, auto_cancel
    
    @property    
    def content_dict(self):
        residual_action, residual_done = self.residual_policy.step()
        content_dict = {
            "type" : 1, # submission of a new limit order
            "direction" : -1 if self.action[0] == 0 else 1,
            "size" : self.action[1] + residual_action,
            "price": self.price,
            "trade_id":self.trade_id,
            "order_id":self.order_id,
            "time":self.time,
        }
        return content_dict
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
        return 0 #TODO: implement
    '''revise it outside the class, (revised in the class Exchange)'''