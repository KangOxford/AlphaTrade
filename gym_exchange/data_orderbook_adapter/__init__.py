# -*- coding: utf-8 -*-
from gym_exchange import Config

class Debugger: 
    on = True # by default
    # on = False
    class Encoder:
        on = True
        # on = False
    class DebugDecoder:
        on = True
        # on = False
        
#
# class Configuration:
#     price_level = 10
#     horizon = Config.raw_horizon
#     # horizon = 4000
#     # horizon = 4096
#     # side_list = ['Bids']
#     class Adapter:
#         type5_id_bid = 30000000  # caution about the volumn for valid numbers
#         type5_id_ask = 40000000  # caution about the volumn for valid numbers
