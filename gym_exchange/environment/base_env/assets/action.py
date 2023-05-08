# ========================== 04 ==========================
import numpy as np
from gym_exchange import Config
from gym_exchange.exchange.basic_exc.assets.order_flow import OrderFlow
from gym_exchange.environment.base_env.assets.price_delta import PriceDelta
from gym_exchange.environment.base_env.assets.initial_policy import ResidualPolicy_Factory


# def singleton(cls):
#     _instance = {}
#     def _singleton(*args, **kwargs):
#         if cls not in _instance:
#             _instance[cls] = cls(*args, **kwargs)
#         return _instance[cls]
#     return _singleton


class IdGenerator():
    '''singleton method
    There should always be only one id generator object'''
    def __init__(self, initial_number):
        self.initial_id = initial_number
        self.current_id = initial_number
    def step(self):
        # print("IdGenerator Step") #$
        self.current_id += 1
        return self.current_id
    
# @singleton
class TradeIdGenerator(IdGenerator):
    def __init__(self, initial_number=Config.trade_id_generator):
        super().__init__(initial_number=initial_number)
        '''type(TradeIdGenerator) : function
        '''
    def step(self):
        super().step()
        # print(f"TradeIdGenerator Step {self.current_id}") #$
        return self.current_id

        
# @singleton
class OrderIdGenerator(IdGenerator):
    def __init__(self, initial_number = Config.order_id_generator):
        super().__init__(initial_number = initial_number)
    def step(self):
        super().step()
        # print(f"OrderIdGenerator Step {self.current_id}") #$
        return self.current_id
        
# ========================== 05 ==========================

class OrderFlowGenerator(object):
    def __init__(self):
        self.residual_policy = ResidualPolicy_Factory.produce("Twap")
        self.trade_id_generator = TradeIdGenerator() 
        self.order_id_generator = OrderIdGenerator()
    
        
    def step(self, action: np.ndarray, best_ask_bid_dict, num_hold, kind = 'limit_order') -> OrderFlow:
        # shoud the price list be one sided or two sided???? #TODO
        self.action = action # [side, quantity_delta, price_delta]
        self.best_ask_bid_dict = best_ask_bid_dict
        self.num_hold = num_hold
        self.kind = kind
        content_dict, revised_content_dict = self.get_content_dicts()
        # [side, quantity_delta, price_delta] => [side, quantity, price]
        order_flow = OrderFlow(**content_dict)
        auto_cancel = OrderFlow(**revised_content_dict) # TODO
        return order_flow, auto_cancel
     
    def get_content_dicts(self):
        if self.kind == "market_order":
            content_dict = {
            "Type" : 0, # submission of a new market order
            "direction" : self.action[0], # TODO masked for oneside task
            "size": min(max(0, self.action[1]), self.num_hold), # 0<=size<=num_hold
            "price": self.price, # call @property: price(self)
            "trade_id":self.trade_id,
            "order_id":self.order_id,
            "time":self.time,
            }
        elif self.kind == 'limit_order':
            self.residual_action, self.residual_done = self.residual_policy.step()
            content_dict = {
                "Type" : 1, # submission of a new limit order
                # "direction" : self.action[0], # TODO should be right
                "direction" : self.action[0], # TODO masked for oneside task
                "size": min(max(0, self.action[1] + self.residual_action[0]), self.num_hold), # 0<=size<=num_hold
                "price": self.price, # call @property: price(self)
                "trade_id":self.trade_id,
                "order_id":self.order_id,
                "time":self.time,
            }
        else: raise NotImplementedError
        print(f"real_action(order_flow):{content_dict}")#$
        if content_dict['price'] == None:
            print()
        '''used for to-be-sumbmitted oreders'''
        revised_content_dict = {
            "Type" : 3, # total deletion of a limit order
            "direction" : content_dict['direction'], # keep the same direction
            "size"      : content_dict['size'],
            "price"     : content_dict['price'],
            "trade_id"  : content_dict['trade_id'],
            "order_id"  : content_dict['order_id'],
            "time"      : content_dict['time'],
        }
        '''used for generating autocancel oreders'''
        assert content_dict['size'] >= 0, "The real quote size should be non-negative"
        return content_dict, revised_content_dict
    

    @property
    def price(self):
        real_action = self.residual_action[1] ^ int(self.action[2])
        # real_action = self.residual_action[1]^ (self.action[2])
        # price_delta initial real_action
        # 0           0       0
        # 0           1       1
        # 1           0       1
        # 1           1       0
        # for price_delta, 0 means keep, 1 means change
        return PriceDelta(self.best_ask_bid_dict)(side = 'ask' if self.action[0]==0 else 'bid', price_delta = real_action) # side, price_delta
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
