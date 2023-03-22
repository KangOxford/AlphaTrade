class Deletion():
    def __init__(self, order_book, message):
        self.order_book = order_book
        self.message = message
    
    @property
    def cancelled_quantity(self): return  min(self.cancel_quantity, self.remaining_quantity)

    def step(self):
        self.order_book.cancel_order(
            side = message['side'], 
            order_id = message['order_id'],
            time = message['timestamp'], 
            # order_id = order.order_id,
            # time = order.timestamp, 
            )
        return self.order_book



class PartDeletion():
    def __init__(self):
        pass
    # def scaling(self):
        '''as part of the tobe partly cancelled orders may already be executed,
           we can only cancel the residual part, with the quantity scalled.
           The scaling rule would be proportion of the [remianing] and [?]'''
    # def 
    
    
class TotalDeletion():
    def __init__(self):
        pass    
    
    
if __name__ =="__main__":
    pass