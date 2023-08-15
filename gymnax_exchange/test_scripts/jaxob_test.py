from functools import partial, partialmethod
from typing import OrderedDict
from jax import numpy as jnp
import jax
import gymnax_exchange.jaxob.JaxOrderBookArrays as job
import time
import timeit


import sys
sys.path.append('/Users/sasrey/AlphaTrade')
import gymnax_exchange


from gym_exchange.data_orderbook_adapter.raw_encoder import RawDecoder, RawEncoder
from gym_exchange.data_orderbook_adapter.decoder import Decoder
from gym_exchange.data_orderbook_adapter.encoder import Encoder
from gym_exchange.data_orderbook_adapter.data_pipeline import DataPipeline




class OrderBook():
    def __init__(self,nOrders=10,nTrades=10) -> None:
        self.nOrders=nOrders
        self.nTrades=nTrades
        self.bids=(jnp.ones((nOrders,6))*-1).astype("int32")
        self.asks=(jnp.ones((nOrders,6))*-1).astype("int32")
        self.trades=(jnp.ones((nTrades,6))*-1).astype("int32")
        

    


    #For loop with jitted functions inside the for loop.
    def process_order_array(self,order_array):
        bidside=self.bids
        askside=self.asks
        trades=self.trades
        ordersides=(askside.astype(jnp.int32),bidside.astype(jnp.int32),trades.astype(jnp.int32))
        for msg in order_array:
            ordersides,_=job.cond_type_side(ordersides,msg)
        self.asks=ordersides[0]
        self.bids=ordersides[1]
        self.trades=ordersides[2]

    def process_order_array_scan(self,order_array):
        bidside=self.bids
        askside=self.asks
        trades=self.trades
        ordersides=job.scan_through_entire_array(order_array,(askside,bidside,trades))
        self.asks=ordersides[0]
        self.bids=ordersides[1]
        self.trades=ordersides[2]
        return ordersides[0],ordersides[1],ordersides[2]


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
        ordersides=(asksides,bidsides,jnp.ones_like(bidsides)*-1)
        print("Msg Batch Shape:",msg_arrays.shape)
        print("Bidside Batch Shape:",bidsides.shape)
        print("Askside Batch Shape:",asksides.shape)
        ordersides=job.vscan_through_entire_array(msg_arrays,ordersides)
        return ordersides


    def get_L2_state_experimental(self,N):
        return job.get_L2_state(N,self.asks,self.bids)

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
     ##Configuration variables:
    ordersPerSide=100
    instancesInParallel=1000
    runfirstcall=True
    run2ndcall=True
    runmult_for=False
    runmult_scan=True
    printTimes=True
    printResults=False
    time_individual_msgs=True
    develop=True
    load_data=True
    if load_data:
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
        for flow_list in flow_lists[0:9]:
            for flow in flow_list:
                jax_list.append(flow.to_list)
                message_list.append(flow.to_message)
        message_array=jnp.array(jax_list)
        #print("Messages processed: \n",message_array)
        #print("1st message: " ,message_array[0])


   


    print("Processing...")
    #Process a single set of messages (message_array) through the orderbook. This is largely to make sure everything is jitted and d
    if runfirstcall:
        ob=OrderBook(nOrders=ordersPerSide,nTrades=10)
        ob.process_order_array_scan(message_array)
        if printResults:
            print(message_array)
            print(ob.trades)
        #ob_branch=OrderBook(nOrders=ordersPerSide)
        #ob_branch.process_order_array_branch(message_array)

    start=time.time()
    #Process a second set of messages in a seperate orderbook object. 
    if run2ndcall:
        ob2=OrderBook(nOrders=ordersPerSide)
        #Include timing
        start=time.time()
        ob2.process_order_array(message_array)   
        #Print the output state of both sides of the orderbook.
        if printResults:
            print("Asks: \n",ob2.asks)
            print("Bids: \n",ob2.bids)
    end_single=time.time()-start


    #Process a set of messages N times in parallel, with a for loop going through messages sequentially but processing all N orderbooks in parallel.
    if runmult_for:
        ob4=OrderBook(nOrders=ordersPerSide)
        start=time.time()
        val=ob4.vprocess_mult_order_arrays(message_array,instancesInParallel)
        if printResults:
            print(jax.tree_util.tree_structure(val))
    end_for=time.time()-start


    if runmult_scan:
        ob3=OrderBook(nOrders=ordersPerSide)
        start=time.time()
        val=ob3.vprocess_mult_order_arrays_scan(message_array,instancesInParallel)
        if printResults:
            print(val)
            print(jax.tree_util.tree_structure(val))
    end_scan=time.time()-start
    
    if printTimes:
        print("Time required for N=",1," single instance with cond:", end_single)
        print("Time for for loop with N=",instancesInParallel," instances with cond in a for loop (vmap inside)", end_for)
        print("Time for scan with N=",instancesInParallel," instances with cond in a scan (vmap outside)", end_scan)

    if time_individual_msgs:
        #test_msg=jnp.array([1,1,10,10,0,0,5,5])
        lim_msg={
        'side':-1,
        'type':1,
        'price':10000,
        'quantity':10,
        'orderid':0,
        'traderid':1,
        'time':34000,
        'time_ns':78}

        cncl_msg={
        'side':1,
        'type':3,
        'price':10000,
        'quantity':10,
        'orderid':0,
        'traderid':1,
        'time':34000,
        'time_ns':90}


        match_msg={
        'side':1,
        'type':1,
        'price':11000,
        'quantity':100,
        'orderid':0,
        'traderid':1,
        'time':34000,
        'time_ns':89897}


        b=(jnp.ones((100,6))*-1).astype(jnp.int32)
        a=(jnp.ones((100,6))*-1).astype(jnp.int32)
        t=(jnp.ones((1000,6))*-1).astype(jnp.int32)

        returnval=job.ask_lim(lim_msg,a,b,t)



        #job.ask_lim(msg,ask,bid,trades)
        n_runs=10000
        print("Limit time:",timeit.timeit('val=job.ask_lim(lim_msg,a,b,t); jax.block_until_ready(val)',number=n_runs,globals=globals())/n_runs)

        a,b,t=returnval
        returnval=job.ask_lim(lim_msg,a,b,t)
        a,b,t=returnval
        returnval=job.ask_lim(lim_msg,a,b,t)
        a,b,t=returnval
        returnval=job.ask_lim(lim_msg,a,b,t)
        a,b,t=returnval
        returnval=job.ask_cancel(cncl_msg,a,b,t)   
        print("Cancel time:",timeit.timeit('val=job.ask_cancel(cncl_msg,a,b,t); jax.block_until_ready(val)',number=n_runs,globals=globals())/n_runs)

        a,b,t=returnval
        returnval=job.bid_lim(match_msg,a,b,t)
        print("Matching time:",timeit.timeit('val=job.bid_lim(match_msg,a,b,t); jax.block_until_ready(val)',number=n_runs,globals=globals())/n_runs)
        #print(returnval)

    if develop:
        print('developing')