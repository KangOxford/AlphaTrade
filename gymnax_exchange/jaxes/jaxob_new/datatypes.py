from functools import partial, partialmethod
from typing import OrderedDict
from jax import numpy as jnp
import jax
import JaxOrderBookArrays as job

import time


import sys
sys.path.append('/Users/sasrey/AlphaTrade')
import gymnax_exchange
import gym_exchange

from gym_exchange.data_orderbook_adapter.raw_encoder import RawDecoder, RawEncoder
from gym_exchange.data_orderbook_adapter.decoder import Decoder
from gym_exchange.data_orderbook_adapter.encoder import Encoder
from gym_exchange.data_orderbook_adapter.data_pipeline import DataPipeline


class OrderInBook():
    def __init__(self,price : jnp.DeviceArray, quantity :jnp.DeviceArray, orderid : jnp.DeviceArray, traderid: jnp.DeviceArray,time:jnp.DeviceArray,auxdata) -> None:
        self.price=price
        self.quantity=quantity
        self.orderid=orderid
        self.traderid=traderid
        self.time=time        

    def __str__(self):
        return "p=% s, q=% s, oid=% s,tid= % s,t=% s" % (self.price, self.quantity,self.orderid,self.traderid,self.time) 
'''
def flatten_orderinbook(tree : OrderInBook):
    """Specifies how to flatten a OrderInBook class object.
    
    Args:
        tree: OrderInBook class object represented as Pytree node
    Returns:
        A pair of an iterable with the children to be flattened recursively,
        and some opaque auxiliary data to pass back to the unflattening recipe.
        The auxiliary data is stored in the treedef for use during unflattening.
        The auxiliary data could be used, e.g., for dictionary keys.
    """
    
    children = (tree.price,tree.quantity,tree.orderid,tree.traderid,tree.time)
    aux_data = None # We don't want to treat the name as a child - so far there are no aux data I want to consider.
    return (children, aux_data)


def unflatten_orderinbook(aux_data, children):
    """Specifies how to unflattening a Counter class object.

    Args:
        aux_data: the opaque data that was specified during flattening of the
            current treedef.
        children: the unflattened children
    Returns:
        A re-constructed object of the registered type, using the specified
        children and auxiliary data.
    """
    return OrderInBook(*children, aux_data)

jax.tree_util.register_pytree_node(
    OrderInBook,
    flatten_orderinbook,    # tell JAX what are the children nodes
    unflatten_orderinbook   # tell JAX how to pack back into a `OrderInBook`
)'''

class OrderBook():
    def __init__(self,nOrders=10) -> None:
        self.nOrders=nOrders
        self.bids=(jnp.ones((nOrders,6))*-1).astype("int32")
        self.asks=(jnp.ones((nOrders,6))*-1).astype("int32")
        

    @partial(jax.jit,static_argnums=(2,3))
    def branch_type_side(self,data,type,side,askside,bidside):
        msg={
        'side':side,
        'type':type,
        'price':data[1],
        'quantity':data[0],
        'orderid':data[3],
        'traderid':data[2],
        'time':data[4],
        'time_ns':data[5]
        }   
        if side==1:
            if type==1:
                #match with asks side
                #add remainder to bids side
                matchtuple=job.match_against_ask_orders(askside,msg["quantity"],msg["price"],jnp.ones((5,5))*-1)
                #^(orderside,qtm,price,trade)
                msg["quantity"]=matchtuple[1]
                bids=job.add_order(bidside,msg)
                return matchtuple[0],bids,matchtuple[3]
            elif type==2:
                #cancel order on bids side
                return askside,job.cancel_order(bidside,msg),jnp.ones((5,5))*-1
            elif type==3:
                #cancel order on bids side
                return askside,job.cancel_order(bidside,msg),jnp.ones((5,5))*-1
            elif type==4:
                msg["price"]=99999999
                matchtuple=job.match_against_ask_orders(askside,msg["quantity"],msg["price"],jnp.ones((5,5))*-1)
                #^(orderside,qtm,price,trade)
                return matchtuple[0],bidside,matchtuple[3]
        else:
            if type==1:
                #match with bids side
                #add remainder to asks side
                matchtuple=job.match_against_bid_orders(bidside,msg["quantity"],msg["price"],jnp.ones((5,5))*-1)
                #^(orderside,qtm,price,trade)
                msg["quantity"]=matchtuple[1]
                asks=job.add_order(askside,msg)
                return asks,matchtuple[0],matchtuple[3]
            elif type==2:
                #cancel order on asks side
                return job.cancel_order(askside,msg),bidside,jnp.ones((5,5))*-1
            elif type==3:
                #cancel order on asks side
                return job.cancel_order(askside,msg),bidside,jnp.ones((5,5))*-1
            elif type==4:
                #set price to 0
                #match with bids side 
                #no need to add remainder
                msg["price"]=0
                matchtuple=job.match_against_bid_orders(bidside,msg["quantity"],msg["price"],jnp.ones((5,5))*-1)
                #^(orderside,qtm,price,trade)
                return askside,matchtuple[0],matchtuple[3]


    #For loop with jitted functions inside the for loop.
    def process_order_array(self,order_array):
        bidside=self.bids
        askside=self.asks
        for msg in order_array:
            askside,bidside,trades=job.cond_type_side(msg,askside.astype("int32"),bidside.astype("int32"))
        self.asks=askside
        self.bids=bidside

    def process_mult_order_arrays(self,single_array):
        bidsides=self.bids
        asksides=self.asks
        for i in jnp.arange(100):
            bidside=self.bids
            askside=self.asks
            for msg in single_array:
                askside,bidside,trades=job.branch_type_side(msg[2:],int(msg[0]),int(msg[1]),askside.astype("int32"),bidside.astype("int32"))
            asksides=jnp.concatenate((asksides,askside),axis=1)
            bidsides=jnp.concatenate((bidsides,bidside),axis=1)
        return (asksides,bidsides)


    def vprocess_mult_order_arrays(self,single_array):
        msg_arrays=jnp.stack([single_array]*50000,axis=2)
        bidsides=jnp.stack([self.bids]*50000,axis=0).astype("int32")
        asksides=jnp.stack([self.asks]*50000,axis=0).astype("int32")
        print("Msg Batch Shape:",msg_arrays.shape)
        print("Bidside Batch Shape:",bidsides.shape)
        print("Askside Batch Shape:",asksides.shape)
        for msg in msg_arrays:
            asksides,bidsides,trades=job.vcond_type_side(msg,asksides,bidsides)
        return (asksides,bidsides)


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
        aux_data = {'nOrders': self.nOrders}  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(**aux_data)


jax.tree_util.register_pytree_node(OrderBook,
                                    OrderBook._tree_flatten,
                                    OrderBook._tree_unflatten)

#Turns flow lists to have side be -1/1 rather than ask/bid
def to_order_flow_lists(flow_lists): 
    '''change side format from bid/ask to 1/-1
    side = -1 if item.side == 'ask' else 1'''
    for flow_list in flow_lists:
        for item in flow_list:
            side = -1 if item.side == 'ask' else 1
            item.side = side
    return flow_lists





if __name__ == "__main__":
    ob=OrderBook(nOrders=100)
    decoder = RawDecoder(**DataPipeline()())
    encoder = RawEncoder(decoder)
    print(decoder.historical_data.shape)
    flow_lists=encoder()
    flow_lists = to_order_flow_lists(flow_lists)
    single_message=flow_lists[0].get_head_order_flow().to_message
    jax_list=[]
    message_list=[]

    for flow_list in flow_lists[0:2000]:
        for flow in flow_list:
            jax_list.append(flow.to_list)
            message_list.append(flow.to_message)

    message_array=jnp.array(jax_list)
    #print("Messages processed: \n",message_array)
    #print("1st message: " ,message_array[0])
    print("Processing...")
    #ob.process_order_array(message_array)
    #ob2=OrderBook(nOrders=100)
    
    #start=time.time()
    #ob2.process_order_array(message_array)   
    
    #print("Asks: \n",ob2.asks)
    #print("Bids: \n",ob2.bids)

    """
    rettuple=ob2.process_mult_order_arrays(message_array)
    print(rettuple)
    print(rettuple[0].shape)
    """ 
    #end=time.time()-start
    #print(end)
    ob3=OrderBook(nOrders=100)
    start=time.time()
    val=ob3.vprocess_mult_order_arrays(message_array)
    print(val)
    end=time.time()-start
    print(end)
    
    #bids,asks=ob.get_L2_state(5)
    #print("Bids: \n",bids)
    #print("Asks: \n",asks)
    #print(ob.get_totquant_at_price(ob.asks,8600.))


    #print(ob.process_mult_order_arrays(message_array))