class ExecutedPairs():
    def __init__(self):
        self.market_pairs = []
        self.agent_pairs  = []
        
    def step(self, trades, kind):
        batch = self.trades2pairs(trades)
        self.update(batch, kind)
        
    def trades2pairs(self, trades):
        return pairs #TODO : implement
    
    def update(self, batch, kind):
        if kind == "market": self.market_pairs += batch
        elif kind=="agent" : self.agent_pairs  += batch
        else: raise NotImplementedError
        
        
""" trades format
transaction_record = {
        'timestamp': self.time,
        'price': traded_price,
        'quantity': traded_quantity,
        'time': self.time
        }
if side == 'bid':
    transaction_record['party1'] = [counter_party, 'bid', head_order.order_id, new_book_quantity]
    transaction_record['party2'] = [quote['trade_id'], 'ask', None, None]
else:
    transaction_record['party1'] = [counter_party, 'ask', head_order.order_id, new_book_quantity]
    transaction_record['party2'] = [quote['trade_id'], 'bid', None, None]
"""

'''pairs format

'''

if __name__ == "__main__":
    pass