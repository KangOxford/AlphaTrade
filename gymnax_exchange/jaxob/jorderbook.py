# import importlib
# from typing import Dict
# import JaxOrderbook as job
# job=importlib.reload(job)
# from jax import numpy as jnp
# from jax import lax
# import collections


import importlib
from os import remove
from readline import remove_history_item
from typing import Dict
from unicodedata import bidirectional
import gymnax_exchange.jaxob.JaxOrderbook as job
job=importlib.reload(job)
from jax import numpy as jnp
from jax import lax




class OrderBook(object):
    def __init__(self, price_levels=100,orderQueueLen=100):
        self.price_levels=price_levels
        orderbookDimension=[2,price_levels,orderQueueLen,job.ORDERSIZE]
        self.orderbook_array=jnp.ones(orderbookDimension)*-1
        self.orderbook_array=self.orderbook_array.astype(int)

    def process_order(self,quote:Dict,from_data=False,verbose=False):
        '''Wrapper function for the object class that takes a Dict Object as the quote,
         ensures the order is conserved and turns the values into a jnp array which is passed to the JNP ProcessOrder function'''
        #Type, Side,quant,price
        inttype=5
        if quote['type']=='limit':
            inttype=1
        elif quote['type']=='cancel':
            inttype=2
        elif quote['type']=='delete':
            inttype=3
        elif quote['type']=='market':
            inttype=4
        intside=-1
        if quote['side']=='bid':
            intside=1
                
        order_array=jnp.array([inttype,intside,quote['quantity'],quote['price'],quote['trade_id'],quote['order_id'],int(quote['timestamp'].split('.')[0]),int(quote['timestamp'].split('.')[1])])
        self.orderbook_array,trades=job.processOrder(self.orderbook_array,order_array)
        return trades,order_array

    def process_orders_array(self,msgs):
        '''Wrapper function for the object class that takes a JNP Array of messages (Shape=Nx8), and applies them, in sequence, to the orderbook'''
        self.orderbook_array,trades=lax.scan(job.processOrder,self.orderbook_array,msgs)
        return trades


    def get_volume_at_price(self, side, price):
        bidAsk=int((side+1)/2)#buy is 1, sell is 0 # side: buy is 1, sell is -1
        idx=jnp.where((self.orderbook_array[bidAsk,:,:,1]==price),size=1,fill_value=-1)
        volume=self.orderbook_array[bidAsk][idx][0,0]
        return volume
    
    def update_time(self):
        '''Not really functional, don't see the point, but copied from the CPU version.'''
        self.time += 1

    def get_best_bid(self):
        return self.orderbook_array[1,0,0,1]

    def get_worst_bid(self):
        '''This is slightly annoying to implement - Not sure what index the worst price will be at'''
        return NotImplementedError

    def get_best_ask(self):
        return self.orderbook_array[0,0,0,1]

    def get_worst_ask(self):
        '''This is slightly annoying to implement - Not sure what index the worst price will be at'''
        return NotImplementedError

    def tape_dump(self, filename, filemode, tapemode):
        '''Not really sure what to do with this'''
        return 0


    def get_L2_state(self):
        levels=jnp.resize(jnp.array([0,self.price_levels*2,self.price_levels,self.price_levels*3]),(1,4*self.price_levels)).squeeze()
        index=jnp.resize(jnp.arange(0,self.price_levels,1),(4,self.price_levels)).transpose().reshape(1,4*self.price_levels).squeeze()
        prices=jnp.squeeze(self.orderbook_array[:,:,0,1])
        volumes=jnp.squeeze(jnp.sum(self.orderbook_array[:,:,:,0].at[self.orderbook_array[:,:,:,0]==-1].set(0),axis=2))
        return jnp.concatenate([prices,volumes]).flatten()[index+levels]
