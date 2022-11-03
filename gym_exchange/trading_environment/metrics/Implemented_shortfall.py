# ========================== 02 ==========================
from gym_exchange.exchange import Config

class ImplementedShortfall():
    def __init__(self):
        pass
    
    def q(self,t):
        return 0 #TODO
    def u(self,t):
        return 0 #TODO
    
    def inventory_cost(self,t):
        temp = self.q(t) - self.u(t)
        inventory_cost = Config.phi_prime * temp * temp
        return inventory_cost
    
    def liuquidating_revenue(self,t):
        u_t = self.u(t)
        return 0 #TODO
    
    
class RelativePerformance_IS():
    def __init__(self):
        pass
    
    
    
if __name__ == "__main__":
    pass    