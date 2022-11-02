# ========================== 01 ==========================

# class Vwap():
#     time_window = 1 # one step
#     def __init__(self, historical_data, running_data):
#         self.historical_data = historical_data
#         self.running_data = running_data
        
#     @property
#     def difference(self):
#         pass
    

    
# class StaticVwap():
#     def __init__(self, historical_data, running_data):
#         pass
    
# class DynamicVwap():
#     def __init__(self):
#         pass



class VwapCurve():
    ''' curve format
        index | 0 | 1 | 2 | 3 | 4 | 5 |
        pirce |0.1|0.2|0.3|0.4|0.5|0.6|'''
    def __init__(self,):
        self.index = 0
        
    def to_array(self):
        return np.array(result)
    

class VwapEstimator():
    def __init__(self):
        pass
    
    def vwap_price(self, pairs):
        ''' pairs format
        price:    array([[ 1. ,  1. ,  1. ,  1.1,  0.9],
        quantity:        [ 2. , 23. ,  3. , 21. ,  3. ]])
        '''
        vwap_price = (pairs[0]*pairs[1]).sum()/pairs[1].sum()
        return vwap_price
    
    @property
    def market_vwap(self):
        return self.vwap_price(self.market_pairs)
    
    @property
    def agent_vwap(self):
        return self.vwap_price(self.agent_pairs)
    
    def update(executed_pairs):
        pass

    

if __name__ == "__main__":
    # vwap_price testing
    import numpy as np
    pairs = np.array([[1,2],[1,23],[1,3],[1.1,21],[0.9,3]]).T
    
    
    
    
    
    
    
    
    
    
    
