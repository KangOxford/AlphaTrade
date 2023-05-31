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
            ordersides,trades=job.cond_type_side((askside.astype("int32"),bidside.astype("int32")),msg)
        self.asks=ordersides[0]
        self.bids=ordersides[1]

    def process_order_array_branch(self,order_array):
        bidside=self.bids
        askside=self.asks
        for msg in order_array:
            ordersides,trades=job.branch_type_side(msg[2:],int(msg[0]),int(msg[1]),askside.astype("int32"),bidside.astype("int32"))
        self.asks=ordersides[0]
        self.bids=ordersides[1]

    def process_order_array_scan(self,order_array):
        bidside=self.bids
        askside=self.asks
        ordersides,trades=job.scan_through_entire_array(order_array,(askside,bidside))
        return ordersides[0],ordersides[1],trades

    def process_mult_order_arrays(self,single_array,Nparallel):
        bidsides=self.bids
        asksides=self.asks
        for i in jnp.arange(Nparallel):
            bidside=self.bids
            askside=self.asks
            for msg in single_array:
                askside,bidside,trades=job.branch_type_side(msg[2:],int(msg[0]),int(msg[1]),askside.astype("int32"),bidside.astype("int32"))
            asksides=jnp.concatenate((asksides,askside),axis=1)
            bidsides=jnp.concatenate((bidsides,bidside),axis=1)
        return (asksides,bidsides)


    def vprocess_mult_order_arrays(self,single_array,Nparallel):
        msg_arrays=jnp.stack([single_array]*Nparallel,axis=2)
        bidsides=jnp.stack([self.bids]*Nparallel,axis=0).astype("int32")
        asksides=jnp.stack([self.asks]*Nparallel,axis=0).astype("int32")
        ordersides=(asksides,bidsides)
        print("Msg Batch Shape:",msg_arrays.shape)
        print("Bidside Batch Shape:",bidsides.shape)
        print("Askside Batch Shape:",asksides.shape)
        for msg in msg_arrays:
            ordersides,trades=job.vcond_type_side(ordersides,msg)
        return ordersides,trades

    def vprocess_mult_order_arrays_scan(self,single_array,Nparallel):
        msg_arrays=jnp.stack([single_array]*Nparallel,axis=2)
        bidsides=jnp.stack([self.bids]*Nparallel,axis=0).astype("int32")
        asksides=jnp.stack([self.asks]*Nparallel,axis=0).astype("int32")
        ordersides=(asksides,bidsides)
        print("Msg Batch Shape:",msg_arrays.shape)
        print("Bidside Batch Shape:",bidsides.shape)
        print("Askside Batch Shape:",asksides.shape)
        ordersides,trades=job.vscan_through_entire_array(msg_arrays,ordersides)
        return ordersides,trades


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
    #Loading data from the LOBSTER dataset using Kang's data encoder - where all the pre-processing is done. 
    decoder = RawDecoder(**DataPipeline()())
    encoder = RawEncoder(decoder)
    flow_lists=encoder()
    #Function by me to change the format of the flow lists slightly.
    flow_lists = to_order_flow_lists(flow_lists)
    #Single message for debugging purposes. 
    single_message=flow_lists[0].get_head_order_flow().to_message
    jax_list=[]
    message_list=[]

    #Creating two types of lists of messages to give to the orderbook:
    # A) jax_list is a list with messages ready for the jax OB (i.e. numerical)
    for flow_list in flow_lists[0:2000]:
        for flow in flow_list:
            jax_list.append(flow.to_list)
            message_list.append(flow.to_message)

    message_array=jnp.array(jax_list)
    #print("Messages processed: \n",message_array)
    #print("1st message: " ,message_array[0])


    ##Configuration variables:
    ordersPerSide=100
    instancesInParallel=10000


    print("Processing...")
    
    #Process a single set of messages (message_array) through the orderbook. This is largely to make sure everything is jitted and d
    ob=OrderBook(nOrders=ordersPerSide)
    ob.process_order_array(message_array)
    ob_branch=OrderBook(nOrders=ordersPerSide)
    ob_branch.process_order_array_branch(message_array)
    
    #Process a second set of messages in a seperate orderbook object. 
    ob2=OrderBook(nOrders=ordersPerSide)
    #Include timing
    start=time.time()
    ob2.process_order_array(message_array)   
    #Print the output state of both sides of the orderbook.
    print("Asks: \n",ob2.asks)
    print("Bids: \n",ob2.bids)
    end_single=time.time()-start

    #Process a second set of messages in a seperate orderbook object under branch
    ob2_branch=OrderBook(nOrders=ordersPerSide)
    #Include timing
    start=time.time()
    ob2_branch.process_order_array_branch(message_array)   
    #Print the output state of both sides of the orderbook.
    print("Asks: \n",ob2.asks)
    print("Bids: \n",ob2.bids)
    end_single_branch=time.time()-start


    #Process a set of messages N times in "parallel" (really in this case it is serial) through the use of two for loops 
    #This uses the branch function (using if-else statements to distinguish message types/side)
    """ob2=OrderBook(nOrders=ordersPerSide)
    start=time.time()
    rettuple=ob2.process_mult_order_arrays(message_array,instancesInParallel)
    print("Output from all orderbooks:",rettuple)
    print(rettuple[0].shape)
    end_for_for=time.time()-start
    """
    

    #Process a set of messages N times in parallel, with a for loop going through messages sequentially but processing all N orderbooks in parallel.
    ob4=OrderBook(nOrders=ordersPerSide)
    start=time.time()
    val=ob4.vprocess_mult_order_arrays(message_array,instancesInParallel)
    print(jax.tree_util.tree_structure(val))
    end_for=time.time()-start



    ob3=OrderBook(nOrders=ordersPerSide)
    start=time.time()
    val=ob3.vprocess_mult_order_arrays_scan(message_array,instancesInParallel)
    print(val)
    print(jax.tree_util.tree_structure(val))
    end_scan=time.time()-start
    
    print("Time required for N=:",1," single instance with cond", end_single)
    print("Time required for N=:",1," single instance with branch", end_single_branch)
    print("Time for for loop with N=",instancesInParallel," instances with cond in a for loop (vmap inside)", end_for)
    print("Time for scan with N=",instancesInParallel," instances with cond in a scan (vmap outside)", end_scan)
    """print("Time required for N=",instancesInParallel," instances with for loops and branch: ",end_for_for)"""

    #bids,asks=ob.get_L2_state(5)
    #print("Bids: \n",bids)
    #print("Asks: \n",asks)
    #print(ob.get_totquant_at_price(ob.asks,8600.))


    #print(ob.process_mult_order_arrays(message_array))