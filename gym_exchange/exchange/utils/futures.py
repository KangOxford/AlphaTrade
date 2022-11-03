from gym_exchange.exchange.order_flow import OrderFlow



class Futures():
    def __init__(self):
        self.data_list = []
    def step(self) -> OrderFlow:
        '''check the order in the data list and see if it is
        waited to be processed at this time step. The step should
        be in the Exchange'''
        return order_flow
    def update(self, auto_cancel: OrderFlow):
        self.data_list.append(auto_cancel)
        
        
        