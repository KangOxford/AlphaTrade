import importlib
from os import remove
from readline import remove_history_item
from typing import Dict
from unicodedata import bidirectional
import gymnax_exchange.jaxob.JaxOrderbook as job
job=importlib.reload(job)
from jax import numpy as jnp
from jax import lax
import jax



class OrderBook(object):
    def __init__(self,nOrderbooks, price_levels=200,orderQueueLen=200):
        self.price_levels=price_levels
        orderbookDimension=[nOrderbooks,2,price_levels,orderQueueLen,job.ORDERSIZE]
        self.orderbooks_array=jnp.ones(orderbookDimension)*-1
        self.orderbooks_array=self.orderbooks_array.astype(int)

    def process_orders_array(self,msgs_batch):
        '''Wrapper function for the object class that takes a JNP Array of messages (Shape=Nx8), and applies them, in sequence, to the orderbook'''
        self.orderbooks_array=job.scanOrders_batch(self.orderbooks_array,msgs_batch)
        return 0
    #No longer really applies in the batched version.
    """
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
"""