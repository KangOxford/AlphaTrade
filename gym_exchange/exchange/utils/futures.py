from gym_exchange.exchange.order_flow import OrderFlow



class Futures():
    def __init__(self):
        self.data_list = []
    def step(self):
        return 0
    def update(self, auto_cancel: OrderFlow):
        self.data_list.append(auto_cancel)
        
        
        