class BaseAction():
    def __init__(self,side,quantity,price_delta,auto_cancel):
        self.side = side
        self.quantity = quantity
        self.price_delta = price_delta
    
    @property
    def to_message(self):
        pass

# -------------------------- 01 ----------------------------
class ComplexAction(BaseAction):
    def __init__(self,side,quantity,price_delta):
        super.__init__(side,quantity,price_delta)
        ''''price_delta = -3,-2,-1,0,1,2,3 
            side = 'bid' or 'ask'
            quantity = 0 ~ num2liquidate (int)
            auto_cancel = 10 # fixed'''
# -------------------------- 02 ----------------------------    
class SideAction(BaseAction):
    def __init__(self,side,quantity):
        super.__init__(side,quantity, price_delta = 0)
        ''''side = 'bid' or 'ask'
            quantity = 0 ~ num2liquidate (int)
            price_delta = 0 # fixed
            auto_cancel = 10 # fixed'''
    
class DeltaAction(BaseAction):
    def __init__(self,quantity,price_delta):
        super.__init__(side,quantity, price_delta, side = 'ask')
        ''''quantity = 0 ~ num2liquidate (int)
            price_delta = -1, 0, 1 
            side = 'ask' # fixed
            auto_cancel = 10 # fixed'''
# -------------------------- 03 ----------------------------    
class SimpleAction(BaseAction):
    def __init__(self,quantity):
        super.__init__(side,quantity, price_delta = 0, side = 'ask')
        ''''quantity = 0 ~ num2liquidate (int)
            side =  'ask' # fixed
            price_delta = 0 # fixed
            auto_cancel = 10 # fixed'''