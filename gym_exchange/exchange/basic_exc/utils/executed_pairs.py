import numpy as np
class ExecutedPairsRecorder():
    def __init__(self):
        self.index = 0
        self.market_pairs = {}
        self.agent_pairs  = {}
        

    def trades2pairs(self, trades): # to be used in step
        pairs = []
        for trade in trades:
            value = np.array([trade['price'], trade['quantity']]).T
            parties = [trade['party1'], trade['party1']]
            for party in parties:
                """ trade_id_generator = 80000000
                    order_id_generator = 88000000 """
                if len(str(party[0])) == 8 and str(party[0])[:2] in ("80","88"): # party[0] is trade id ,Not sure, perhpas order id
                    kind = 'agent'
                else: kind = 'market'
                pair = {kind:value}
                pairs.append(pair)
        return pairs

    def update(self, pairs): # to be used in step
        for pair in pairs:
            for key,value in pair.items(): # Pseudo for loop, one pair dict
                if   key == "market": self.market_pairs[self.index] = self.market_pairs.get(self.index, []) + [value]
                elif key == "agent" : self.agent_pairs[self.index]  = self.agent_pairs.get(self.index, [])  + [value]
                else: raise NotImplementedError

    def step(self, trades, index):
        """two function:
        01: record market pairs and agent pairs, e.g.
        [In]  self.market_pairs
        [Out] {86: array([[31179100],
                        [       9]])}
        02: record the last_executed_pairs of market_agent"""
        # ----------- 01 ------------
        self.index = index # keep the same with the exchange index
        if len(trades) == 0: 
            pass
        else: # len(trades) == 1 or 3
            pairs = self.trades2pairs(trades)
            self.update(pairs)
        # ----------- 02 ------------
        try:
            self.market_agent_executed_pairs_in_last_step = [
                self.index,
                [self.market_pairs[self.index],
                 self.agent_pairs[self.index]]
                ]
        except:
            self.market_agent_executed_pairs_in_last_step = [
                self.index, None]
        
    def __str__(self):
        fstring = f'>>> market_pairs: {self.market_pairs}, \n>>> agent_pairs : {self.agent_pairs}'
        return fstring
        
        
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

''' pairs format
price:    array([[ 1. ,  1. ,  1. ,  1.1,  0.9],
quantity:        [ 2. , 23. ,  3. , 21. ,  3. ]])
'''

'''trade format
{'timestamp': '34201.40462348', 'price': 31180000, 'quantity': 1, 'time': '34201.40462348', 
'party1': [3032093, 'ask', 3032093, None], 
'party2': [15750757, 'bid', None, None]}
'''

if __name__ == "__main__":
    pass
