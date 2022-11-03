from gym_exchange.exchange.order_flow import OrderFlow
class Future():
    def __init__(self,  auto_cancel: OrderFlow):
        self.flow = auto_cancel

    @property
    def maturity(self) -> bool:
        '''determined by the price and mid-price or the exchange'''
        return False   
        

class Futures():
    def __init__(self):
        self.data_list = []
    def step(self) -> OrderFlow:
        '''check the order in the data list and see if it is
        waited to be processed at this time step. The step should
        be in the Exchange'''
        order_flow_list = []
        for future in self.data_list:
            if future.maturity == True:
                order_flow_list.append(future.flow)
        return order_flow_list
    
    def update(self, auto_cancel: OrderFlow):
        future = Future(auto_cancel)
        self.data_list.append(future)
        
        
        