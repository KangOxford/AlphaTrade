# -*- coding: utf-8 -*-

class Debugger: 
    on = True # by default
    # on = False
    
class Configuration:
    price_level = 10 
    horizon = 2048
    # horizon = 4048
    side_list = ['Bids']
    class Adapter:
        type5_id_bid = 30000000  # caution about the volumn for valid numbers
        type5_id_ask = 40000000  # caution about the volumn for valid numbers