class BaseAction():
    def __init__(self,side,quantity,price_delta):
        self.side = side
        self.quantity = quantity
        self.price_delta = price_delta
    
    @property
    def to_message(self):
        pass
    
    # def __str__(self):
    #     fstring = 'side: {self.side}, quantity: {self.quantity}, price_delta: {self.price_delta}'
    #     return fstring


# -------------------------- 01 ----------------------------    
class SideAction(BaseAction):
    def __init__(self,side,quantity):
        super.__init__(side, quantity, price_delta = 0)
        ''''side = 'bid' or 'ask'
            quantity = 0 ~ num2liquidate (int)
            price_delta = 0 # fixed
            auto_cancel = 10 # fixed'''
    
class DeltaAction(BaseAction):
    def __init__(self,quantity,price_delta):
        super.__init__(side, quantity, price_delta, side = 'ask')
        ''''quantity = 0 ~ num2liquidate (int)
            price_delta = -1, 0, 1 
            side = 'ask' # fixed
            auto_cancel = 10 # fixed'''
# -------------------------- 02 ----------------------------    
class SimpleAction(BaseAction):
    def __init__(self,quantity):
        super.__init__(side, quantity, price_delta = 0, side = 'ask')
        ''''quantity = 0 ~ num2liquidate (int)
            side =  'ask' # fixed
            price_delta = 0 # fixed
            auto_cancel = 10 # fixed'''

class PriceDelta():
    def __init__(self, price_delta):
        self.price_delta = price_delta
        
 # =================================================================
        
class Action(BaseAction):
    def __init__(self,side,quantity,price_delta):
        self.side = side 
        self.quantity = quantity
        self.price_delta = price_delta
        # ''''price_delta = -3,-2,-1,0,1,2,3 (0 ~ 7)
        #     side = 'bid'(0) or 'ask'(1)
        #     residual_quantity = -num2liquidate ~ num2liquidate (int: 0 ~ 2*num2liquidate+1)''' 