from gym_exchange import Config
from gym_exchange.exchange.basic_exc.assets.order_flow import OrderFlow
from copy import deepcopy

# class CancellationDeterminants():
#     def __init__(self): 
#         pass
    
#     @property
#     def order_flow_imbalance(self):
#         return float
    
#     @property
#     def price_trend(self):
#         Config.price_trend_window 
#         return float
    
#     def __call__(self):
#         return {
#             "order_flow_imbalance" : self.order_flow_imbalance,
#             "price_trend" : self.price_trend,
#         }

class Timeout():
    def __init__(self): 
        self.time = 0
    
    def step(self):
        self.time += 1
    
    def __call__(self) -> bool:
        return self.time > Config.timeout

class AutoCancel():
    def __init__(self, flow: OrderFlow):
        self.flow = flow
        self.timeout = Timeout()

    def step(self):
        self.timeout.step()

    @property
    def matured(self) -> bool:
        # '''determined by the price and mid-price or the exchange'''
        # '''determined by the quote age'''
        if self.timeout: return True
        else: return False


        
class AutoCancels():
    def __init__(self):
        self.auto_cancels = []
        
    def step(self) -> OrderFlow:
        '''check the order in the data list and see if it is
        waited to be processed at this time step. The step should
        be in the Exchange'''
        for auto_cancel in self.auto_cancels:
            auto_cancel.step()

        order_flow_list = []
        index_list = []
        for index, auto_cancel in enumerate(self.auto_cancels):
            if auto_cancel.matured:
                order_flow_list.append(auto_cancel.flow)
                index_list.append(index)
        for index in index_list:
            # self.auto_cancels.remove(auto_cancel)
            self.auto_cancels.pop(index)
        return order_flow_list
    
    def add(self, auto_cancel: OrderFlow):
        auto_cancel = AutoCancel(auto_cancel)
        self.auto_cancels.append(auto_cancel)
        
        
if __name__ == '__main__':
    auto_cancels = AutoCancels()        
