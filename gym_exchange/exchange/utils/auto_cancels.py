from gym_exchange import Config
from gym_exchange.exchange.order_flow import OrderFlow

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
        self.age = 0
    
    def step(self):
        self.age += 1
    
    def __call__(self) -> bool:
        return self.age > Config.timeout

class CancellationDeterminants():
            
    def __init__(self):
        self.timeout = Timeout()


class AutoCancel():
    def __init__(self,  auto_cancel: OrderFlow):
        self.auto_cancel = auto_cancel
        self.cancellation_determinants = CancellationDeterminants()

    @property
    def maturity(self) -> bool:
        # '''determined by the price and mid-price or the exchange'''
        # '''determined by the quote age'''
        determinant = self.cancellation_determinants.timeout
        if determinant() == False:  return False # if all satisfied then return mature
        else: return True  

        
class AutoCancels():
    def __init__(self):
        self.auto_cancels = []
        
    def step(self) -> OrderFlow:
        '''check the order in the data list and see if it is
        waited to be processed at this time step. The step should
        be in the Exchange'''
        for auto_cancel in self.auto_cancels:
            auto_cancel.cancellation_determinants.timeout.step()
            # try: #$
            #     auto_cancel.cancellation_determinants.timeout.step()
            # except:
            #     breakpoint()
            #     print()#$

        order_flow_list = []
        for auto_cancel in self.auto_cancels:
            if auto_cancel.maturity == True:
                order_flow_list.append(auto_cancel.flow); self.auto_cancels.remove(auto_cancel)
        return order_flow_list
    
    def add(self, auto_cancel: OrderFlow):
        auto_cancel = AutoCancel(auto_cancel)
        self.auto_cancels.append(auto_cancel)
        
        
if __name__ == '__main__':
    auto_cancels = AutoCancels()        
        