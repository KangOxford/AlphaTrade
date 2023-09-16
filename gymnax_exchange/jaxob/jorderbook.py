import importlib
from os import remove
from readline import remove_history_item
from typing import Dict
from unicodedata import bidirectional
import gymnax_exchange.jaxob.JaxOrderBookArrays as job
job=importlib.reload(job)
from jax import numpy as jnp
from jax import lax
import jax



class OrderBook(object):
    def __init__(self, nOrders=100,nTrades=100):
        self.nOrders=nOrders
        self.nTrades=nTrades
        self.bids=(jnp.ones((nOrders,6))*-1).astype(jnp.int32)
        self.asks=(jnp.ones((nOrders,6))*-1).astype(jnp.int32)
        self.trades=(jnp.ones((nTrades,6))*-1).astype(jnp.int32)

    def process_order(self,quote:Dict,from_data=False,verbose=False):
        '''Wrapper function for the object class that takes a Dict Object as the quote,
         ensures the order is conserved and turns the values into a jnp array which is passed to the JNP ProcessOrder function'''
        #Type, Side,quant,price
        inttype=5
        intside=-1
        if quote['side']=='bid':
            intside=1 

        if quote['type']=='limit':
            inttype=1
        elif quote['type']=='cancel':
            inttype=2
        elif quote['type']=='delete':
            inttype=2
        elif quote['type']=='market':
            inttype=1
            intside=intside*-1

             
        msg=jnp.array([inttype,intside,quote['quantity'],quote['price'],quote['trade_id'],quote['order_id'],int(quote['timestamp'].split('.')[0]),int(quote['timestamp'].split('.')[1])])
        bidside=self.bids
        askside=self.asks
        trades=self.trades
        ordersides=(askside.astype(jnp.int32),bidside.astype(jnp.int32),trades.astype(jnp.int32))
        ordersides,_=job.cond_type_side(ordersides,msg)
        return ordersides[0],ordersides[1],ordersides[2]

    def process_order_array(self,quote:Dict,from_data=False,verbose=False):
        '''Same as process_order but quote is an array.'''
        bidside=self.bids
        askside=self.asks
        trades=self.trades
        ordersides=(askside.astype(jnp.int32),bidside.astype(jnp.int32),trades.astype(jnp.int32))
        ordersides,_=job.cond_type_side(ordersides,quote)
        return ordersides[0],ordersides[1],ordersides[2]


    def process_orders_array(self,msgs):
        '''Wrapper function for the object class that takes a JNP Array of messages (Shape=Nx8), and applies them, in sequence, to the orderbook'''
        bidside=self.bids
        askside=self.asks
        trades=self.trades
        ordersides=job.scan_through_entire_array(msgs,(askside,bidside,trades))
        self.asks=ordersides[0]
        self.bids=ordersides[1]
        self.trades=ordersides[2]
        return ordersides[0],ordersides[1],ordersides[2]



    def get_volume_at_price(self, side, price):
        '''Need to give the actual askside or bidside object as a parameter to the function, even if they're stored in self'''
        volume=jnp.sum(jnp.where(side[:,0]==price,side[:,1],0))
        return volume
    

    def get_best_bid(self):
        return job.get_best_bid(self.asks,self.bids)

    def get_worst_bid(self):
        '''This is slightly annoying to implement - Not sure what index the worst price will be at'''
        return NotImplementedError

    def get_best_ask(self):
        return job.get_best_ask(self.asks,self.bids)

    def get_worst_ask(self):
        '''This is slightly annoying to implement - Not sure what index the worst price will be at'''
        return NotImplementedError

    def tape_dump(self, filename, filemode, tapemode):
        '''Not really sure what to do with this'''
        return 0


    def get_L2_state(self,N):
        bid_prices=jnp.sort(self.bids[:,0])
        ask_prices=jnp.sort(self.asks[:,0])
        topNbid=bid_prices[-N:][::-1]
        index=int(jnp.where(ask_prices[::-1]==-1,fill_value=0,size=1)[0])
        topNask=jnp.concatenate((ask_prices[-index:],jnp.zeros(N).astype("int32")))
        topNask=topNask[0:N]
        bids=jnp.stack((job.get_totquant_at_prices(self.bids,topNbid),topNbid))
        asks=jnp.stack((job.get_totquant_at_prices(self.asks,topNask),topNask))
        return bids.T,asks.T
    
        #Flatten and Unflatten functions so that methods can be appropriately jitted. 
    def _tree_flatten(self):
        children = (self.bids,self.asks)  # arrays / dynamic values
        aux_data = {'nOrders': self.nOrders,'nTrades':self.nTrades}  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(**aux_data)


jax.tree_util.register_pytree_node(OrderBook,
                                    OrderBook._tree_flatten,
                                    OrderBook._tree_unflatten)
