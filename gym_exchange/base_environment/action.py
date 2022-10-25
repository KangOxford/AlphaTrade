class BaseAction:
    def __init__(self,):
        pass
    
    @property
    def to_message(self):
        pass

# -------------------------- 01 ----------------------------
class Action:
    price_delta = -3,-2,-1,0,1,2,3 
    side = 'bid' or 'ask'
    quantity = 0 ~ num2liquidate (int)
    auto_cancel = 10 # fixed
# -------------------------- 02 ----------------------------    
class Action:
    side = 'bid' or 'ask'
    quantity = 0 ~ num2liquidate (int)
    price_delta = 0 # fixed
    auto_cancel = 10 # fixed
    
class Action:
    quantity = 0 ~ num2liquidate (int)
    price_delta = -1, 0, 1 
    side = 'bid' # fixed
    auto_cancel = 10 # fixed
# -------------------------- 03 ----------------------------    
class Action:
    quantity = 0 ~ num2liquidate (int)
    side =  'ask' # fixed
    price_delta = 0 # fixed
    auto_cancel = 10 # fixed