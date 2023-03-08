from gym_exchange import Config
from gym_exchange.trading_environment.basic_env.interface_env import SpaceParams

# ========================== 01 ==========================
class PriceDelta():
    def __init__(self, price_list):
        self.price_list = price_list
    def __call__(self, side, price_delta):
        self.side = side
        delta = self.adjust_price_delta(price_delta)
        if delta == 0: return self.price_list[0] # best_bid or best_ask # TODO: test
        elif delta> 0: return self.positive_modifying(delta)
        else         : return self.negetive_modifying(delta)
    def adjust_price_delta(self, price_delta):
        return price_delta - SpaceParams.Action.price_delta_size_one_side
    def negetive_modifying(self, delta):
        tick_size = Config.tick_size
        if self.side == 'bid': return self.price_list[0] + delta * tick_size
        elif self.side=='ask': return self.price_list[0] - delta * tick_size
    def positive_modifying(self, delta):
        return self.price_list[delta] # delta_th best bid/ best ask
