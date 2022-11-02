import numpy as np
class OrderFlow():
    length = 7
    def __init__(self,Type,direction,size,price,trade_id,order_id,time):
        '''Parameters
        -------------
        time : str
        Type : int
            1: submission of a new limit order
            2: cancellation(partial deletion)
            3: deletion(total deletion)
        order_id : int
        size : int
        price : int
        direction : int
            -1 : limit order at ask side
            +1 : limit order at sell side
        trade_id : int
        '''
        self.type = Type
        self.side = direction
        self.quantity = size
        self.price = price
        self.trade_id = trade_id
        self.order_id = order_id
        self.timestamp = time
    def __call__(self):
        return np.array([
        self.type, 
        self.side, 
        self.quantity, 
        self.price, 
        self.trade_id, 
        self.order_id, 
        self.timestamp 
            ])
    
    # def __str___(self):
