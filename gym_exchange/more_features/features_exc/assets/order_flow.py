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
        self.next_order_flow = None
        self.prev_order_flow = None
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
    # helper functions to get Orders in linked list
    def next_order_flow(self):
        return self.next_order_flow

    def prev_order_flow(self):
        return self.prev_order_flow
    
    def __str__(self):
        return "‖ Type {:2s} | Side {:2s}| Quantity {:3s}| Price {:8s} | Order_ID {:9s}| Time {:15s} ‖".format( \
        str(self.type), str(self.side), str(self.quantity),\
        str(self.price), str(self.order_id), str(self.timestamp))
        # return "Type {:5d}  |  Side {:8d}  |  Quantity {:8d}  |  Price {:8d}  |  Order_ID {:10d}  |  Time {:15s}".format(self.type, self.side, self.quantity, self.price, self.order_id, self.timestamp)
        # here side should be -1 or 1
        
    @property
    def to_message(self):
        '''message = 
        {'type': 'limit','side': bid or ask,'quantity': quantity,\
         'price': price,'trade_id': trade_id, "timestamp":timestamp,\
         'order_id':order_id}'''
        message = {
            'type'     : 'limit',
            'side'     : 'bid' if self.side==1 else 'ask',
            'quantity' : self.quantity,
            'price'    : self.price,
            'trade_id' : self.trade_id,
            'order_id' : self.order_id,
            'timestamp': self.timestamp
            }
        return message