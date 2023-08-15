# from jax import config
# config.update("jax_enable_x64",True)


from textwrap import fill
from typing import OrderedDict
from jax import numpy as jnp
import jax
from functools import partial, partialmethod

#INITID=-9000
#MAXPRICE=999999999
#TODO: Get rid of these magic numbers by allowing a config dict to be passed through as a static arg 

############### ADD AND REMOVE ###############

@jax.jit
def add_order(orderside,msg):
    emptyidx=jnp.where(orderside==-1,size=1,fill_value=-1)[0]
    orderside=orderside.at[emptyidx,:].set(jnp.array([msg['price'],jnp.maximum(0,msg['quantity']),msg['orderid'],msg['traderid'],msg['time'],msg['time_ns']])).astype(jnp.int32)
    return removeZeroNegQuant(orderside)

@jax.jit
def removeZeroNegQuant(orderside):
    return jnp.where((orderside[:,1]<=0).reshape((orderside.shape[0],1)),x=(jnp.ones(orderside.shape)*-1).astype(jnp.int32),y=orderside)


@jax.jit
def cancel_order(orderside,msg):
    # jax.debug.breakpoint()
    condition=((orderside[:,2]==msg['orderid']) | ((orderside[:,0]==msg['price']) & (orderside[:,2]<=-9000)))
    idx=jnp.where(condition,size=1,fill_value=-1)[0]
    orderside=orderside.at[idx,1].set(orderside[idx,1]-msg['quantity'])
    return removeZeroNegQuant(orderside)


############### MATCHING FUNCTIONS ###############

@jax.jit
def match_order(data_tuple):
    orderside,qtm,price,top_order_idx,trade,agrOID,time,time_ns=data_tuple
    newquant=jnp.maximum(0,orderside[top_order_idx,1]-qtm) #Could theoretically be removed as an operation because the removeZeroQuand func also removes negatives. 
    qtm=qtm-orderside[top_order_idx,1]
    qtm=qtm.astype(jnp.int32)
    emptyidx=jnp.where(trade==-1,size=1,fill_value=-1)[0]
    trade=trade.at[emptyidx,:].set(jnp.array([orderside[top_order_idx,0],orderside[top_order_idx,1]-newquant,orderside[top_order_idx,2],[agrOID],[time],[time_ns]]).transpose())
    orderside=removeZeroNegQuant(orderside.at[top_order_idx,1].set(newquant))
    top_order_idx=get_top_bid_order_idx(orderside)
    return (orderside.astype(jnp.int32),jnp.squeeze(qtm),price,top_order_idx,trade,agrOID,time,time_ns)

@jax.jit
def get_top_bid_order_idx(orderside):
    maxPrice=jnp.max(orderside[:,0],axis=0)
    times=jnp.where(orderside[:,0]==maxPrice,orderside[:,4],999999999)
    minTime_s=jnp.min(times,axis=0)
    times_ns=jnp.where(times==minTime_s,orderside[:,5],999999999)
    minTime_ns=jnp.min(times_ns,axis=0)
    return jnp.where(times_ns==minTime_ns,size=1,fill_value=-1)[0]

@jax.jit
def check_before_matching_bid(data_tuple):
    orderside,qtm,price,top_order_idx,trade,_,_,_=data_tuple
    returnarray=(orderside[top_order_idx,0]>=price) & (qtm>0) & (orderside[top_order_idx,0]!=-1)
    return jnp.squeeze(returnarray)

@jax.jit
def match_against_bid_orders(orderside,qtm,price,trade,agrOID,time,time_ns):
    top_order_idx=get_top_bid_order_idx(orderside)
    orderside,qtm,price,top_order_idx,trade,_,_,_=jax.lax.while_loop(check_before_matching_bid,match_order,(orderside,qtm,price,top_order_idx,trade,agrOID,time,time_ns))
    return (orderside,qtm,price,trade)

@jax.jit
def check_before_matching_ask(data_tuple):
    orderside,qtm,price,top_order_idx,trade,_,_,_=data_tuple
    returnarray=(orderside[top_order_idx,0]<=price) & (qtm>0) & (orderside[top_order_idx,0]!=-1)
    return jnp.squeeze(returnarray)

@jax.jit
def match_against_ask_orders(orderside,qtm,price,trade,agrOID,time,time_ns):
    top_order_idx=get_top_ask_order_idx(orderside)
    orderside,qtm,price,top_order_idx,trade,_,_,_=jax.lax.while_loop(check_before_matching_ask,match_order,(orderside,qtm,price,top_order_idx,trade,agrOID,time,time_ns))
    return (orderside,qtm,price,trade)

@jax.jit
def get_top_ask_order_idx(orderside):
    prices=orderside[:,0]
    prices=jnp.where(prices==-1,999999999,prices)
    minPrice=jnp.min(prices)
    times=jnp.where(orderside[:,0]==minPrice,orderside[:,4],999999999)
    minTime_s=jnp.min(times,axis=0)
    times_ns=jnp.where(times==minTime_s,orderside[:,5],999999999)
    minTime_ns=jnp.min(times_ns,axis=0)
    return jnp.where(times_ns==minTime_ns,size=1,fill_value=-1)[0]

############### MAIN BRANCHING FUNCS ###############

@jax.jit
def cond_type_side(ordersides,data):
    askside,bidside,trades=ordersides
    #jax.debug.breakpoint()
    #jax.debug.print("Askside before is \n {}",askside)
    msg={
    'side':data[1],
    'type':data[0],
    'price':data[3],
    'quantity':data[2],
    'orderid':data[5],
    'traderid':data[4],
    'time':data[6],
    'time_ns':data[7]}
    index=((msg["side"]+1)*2+msg["type"]).astype(jnp.int32)
    ask,bid,trade=jax.lax.switch(index-1,(ask_lim,ask_cancel,ask_cancel,ask_mkt,bid_lim,bid_cancel,bid_cancel,bid_mkt),msg,askside,bidside,trades)
    #jax.debug.print("Askside after is \n {}",ask)
    return (ask,bid,trade),0

@jax.jit
def cond_type_side_save_states(ordersides,data):
    askside,bidside,trades=ordersides
    #jax.debug.breakpoint()
    #jax.debug.print("Askside before is \n {}",askside)
    msg={
    'side':data[1],
    'type':data[0],
    'price':data[3],
    'quantity':data[2],
    'orderid':data[5],
    'traderid':data[4],
    'time':data[6],
    'time_ns':data[7]}
    index=((msg["side"]+1)*2+msg["type"]).astype(jnp.int32)
    ask,bid,trade=jax.lax.switch(index-1,(ask_lim,ask_cancel,ask_cancel,ask_mkt,bid_lim,bid_cancel,bid_cancel,bid_mkt),msg,askside,bidside,trades)
    #jax.debug.print("Askside after is \n {}",ask)
    return (ask,bid,trade),(ask,bid,trade)

@jax.jit
def cond_type_side_save_bidask(ordersides,data):
    askside,bidside,trades=ordersides
    #jax.debug.breakpoint()
    #jax.debug.print("Askside before is \n {}",askside)
    msg={
    'side':data[1],
    'type':data[0],
    'price':data[3],
    'quantity':data[2],
    'orderid':data[5],
    'traderid':data[4],
    'time':data[6],
    'time_ns':data[7]}
    index=((msg["side"]+1)*2+msg["type"]).astype(jnp.int32)
    ask,bid,trade=jax.lax.switch(index-1,(ask_lim,ask_cancel,ask_cancel,ask_mkt,bid_lim,bid_cancel,bid_cancel,bid_mkt),msg,askside,bidside,trades)
    #jax.debug.print("Askside after is \n {}",ask)
    # jax.debug.breakpoint()
    return (ask,bid,trade),get_best_bid_and_ask_inclQuants(ask,bid)

vcond_type_side=jax.vmap(cond_type_side,((0,0,0),0))

############### SCAN FUNCTIONS ###############

def scan_through_entire_array(msg_array,ordersides):
    ordersides,_=jax.lax.scan(cond_type_side,ordersides,msg_array)
    return ordersides

def scan_through_entire_array_save_states(msg_array,ordersides,steplines):
    #Will return the states for each of the processed messages, but only those from data to keep array size constant, and enabling variable #of actions (AutoCancel)
    last,all=jax.lax.scan(cond_type_side_save_states,ordersides,msg_array)
    return (all[0][-steplines:],all[1][-steplines:],last[2])

def scan_through_entire_array_save_bidask(msg_array,ordersides,steplines):
    #Will return the states for each of the processed messages, but only those from data to keep array size constant, and enabling variable #of actions (AutoCancel)
    last,all=jax.lax.scan(cond_type_side_save_bidask,ordersides,msg_array)
    # jax.debug.breakpoint()
    return (last[0],last[1],last[2],all[0][-steplines:],all[1][-steplines:])

vscan_through_entire_array=jax.vmap(scan_through_entire_array,(2,(0,0,0)),0)

################ GET CANCEL MESSAGES ################

#Obtain messages to cancel based on a given ID to lookup. Currently only used in the execution environment.
def get_size(bookside,agentID):
    return jnp.sum(jnp.where(bookside[:,3]==agentID,1,0)).astype(jnp.int32)

def getCancelMsgs(bookside,agentID,size,side):
    bookside=jnp.concatenate([bookside,jnp.zeros((1,6),dtype=jnp.int32)],axis=0)
    indeces_to_cancel=jnp.where(bookside[:,3]==agentID,size=size,fill_value=-1)
    cancel_msgs=jnp.concatenate([jnp.ones((1,size),dtype=jnp.int32)*2,\
                                 jnp.ones((1,size),dtype=jnp.int32)*side, \
                                bookside[indeces_to_cancel,1], \
                                bookside[indeces_to_cancel,0], \
                                bookside[indeces_to_cancel,3], \
                                bookside[indeces_to_cancel,2], \
                                bookside[indeces_to_cancel,4], \
                                bookside[indeces_to_cancel,5]],axis=0).transpose()
    return cancel_msgs


#TODO Currently not implemented: should be used for a less naive version of the autocancel
def getCancelMsgs_smart(bookside,agentID,size,side,action_msgs):
    cond=jnp.stack([bookside[:,3]==agentID]*6,axis=1)
    indeces_to_cancel=jnp.where(bookside[:,3]==agentID,size=size,fill_value=0)
    cancel_msgs=jnp.concatenate([jnp.ones((1,size),dtype=jnp.int32)*2, \
                                jnp.ones((1,size),dtype=jnp.int32)*side, \
                                bookside[indeces_to_cancel,1], \
                                bookside[indeces_to_cancel,0], \
                                bookside[indeces_to_cancel,3], \
                                bookside[indeces_to_cancel,2], \
                                bookside[indeces_to_cancel,4], \
                                bookside[indeces_to_cancel,5]],axis=0).transpose()
    cancel_msgs=jnp.where(cancel_msgs==-1,0,cancel_msgs)
    jax.lax.scan(remove_cnl_if_renewed,cancel_msgs,action_msgs)
    return cancel_msgs

#TODO Currently not implemented: should be used for a less naive version of the autocancel
def remove_cnl_if_renewed(cancel_msgs,action_msg):
    jnp.where(cancel_msgs[:,3]==action_msg[3],)
    return cancel_msgs

   

########Type Functions#############

def doNothing(msg,askside,bidside,trades):
    return askside,bidside,trades

def bid_lim(msg,askside,bidside,trades):
    #match with asks side
    #add remainder to bids side
    matchtuple=match_against_ask_orders(askside,msg["quantity"],msg["price"],trades,msg['orderid'],msg["time"],msg["time_ns"])
    #^(orderside,qtm,price,trade)
    msg["quantity"]=matchtuple[1]
    bids=add_order(bidside,msg)
    return matchtuple[0],bids,matchtuple[3]

def bid_cancel(msg,askside,bidside,trades):
    return askside,cancel_order(bidside,msg),trades

def bid_mkt(msg,askside,bidside,trades):
    msg["price"]=999999999
    matchtuple=match_against_ask_orders(askside,msg["quantity"],msg["price"],trades,msg['orderid'],msg["time"],msg["time_ns"])
    #^(orderside,qtm,price,trade)
    return matchtuple[0],bidside,matchtuple[3]


def ask_lim(msg,askside,bidside,trades):
    #match with bids side
    #add remainder to asks side
    matchtuple=match_against_bid_orders(bidside,msg["quantity"],msg["price"],trades,msg['orderid'],msg["time"],msg["time_ns"])
    #^(orderside,qtm,price,trade)
    msg["quantity"]=matchtuple[1]
    asks=add_order(askside,msg)
    return asks,matchtuple[0],matchtuple[3]

def ask_cancel(msg,askside,bidside,trades):
    return cancel_order(askside,msg),bidside,trades

def ask_mkt(msg,askside,bidside,trades):
    #set price to 0
    #match with bids side 
    #no need to add remainder
    msg["price"]=0
    matchtuple=match_against_bid_orders(bidside,msg["quantity"],msg["price"],trades,msg['orderid'],msg["time"],msg["time_ns"])
    #^(orderside,qtm,price,trade)
    return askside,matchtuple[0],matchtuple[3]



######Helper functions for getting information #######


def get_totquant_at_price(orderside,price):
        return jnp.sum(jnp.where(orderside[:,0]==price,orderside[:,1],0))

get_totquant_at_prices=jax.vmap(get_totquant_at_price,(None,0),0)

@partial(jax.jit,static_argnums=0)
def get_L2_state(N,asks,bids):
    bid_prices=-jnp.unique(-bids[:,0],size=N,fill_value=1)
    ask_prices=jnp.unique(jnp.where(asks[:,0]==-1,999999999,asks[:,0]),size=N,fill_value=999999999)
    ask_prices=jnp.where(ask_prices==999999999,-1,ask_prices)

    bid_quants=get_totquant_at_prices(bids,bid_prices)
    ask_quants=get_totquant_at_prices(asks,ask_prices)
    bid_quants=jnp.where(bid_quants<0,0,bid_quants)
    ask_quants=jnp.where(ask_quants<0,0,ask_quants)
    return jnp.stack((ask_prices,ask_quants,bid_prices,bid_quants),axis=1,dtype=jnp.int32)

def get_best_bid_and_ask(asks,bids):
    best_ask=jnp.min(jnp.where(asks[:,0]==-1,999999999,asks[:,0]))
    best_bid=jnp.max(bids[:,0])
    # jax.debug.print("-----")
    # jax.debug.print("best_bid from [get_best_bid_and_ask] {}", best_bid)
    # jax.debug.print("bids {}", bids)
    return best_ask,best_bid

def get_best_bid_and_ask_inclQuants(asks,bids):
    best_ask,best_bid=get_best_bid_and_ask(asks,bids)
    best_ask_Q=jnp.sum(jnp.where(asks[:,0]==best_ask,asks[:,1],0))                     
    best_bid_Q=jnp.sum(jnp.where(bids[:,0]==best_bid,bids[:,1],0))
    best_ask=jnp.array([best_ask,best_ask_Q],dtype=jnp.int32)             
    best_bid=jnp.array([best_bid,best_bid_Q],dtype=jnp.int32)             
    return best_ask,best_bid  


@partial(jax.jit,static_argnums=0)
def init_orderside(nOrders=100):
    return (jnp.ones((nOrders,6))*-1).astype(jnp.int32)


#TODO: Actually complete this function to not only return dummy vars
def get_initial_orders(bookData,idx_window,time):
    orderbookLevels=10
    initid=-9000
    data=jnp.array(bookData[idx_window]).reshape(int(10*2),2)
    newarr = jnp.zeros((int(orderbookLevels*2),8),dtype=jnp.int32)
    initOB = newarr \
        .at[:,3].set(data[:,0]) \
        .at[:,2].set(data[:,1]) \
        .at[:,0].set(1) \
        .at[0:orderbookLevels*4:2,1].set(-1) \
        .at[1:orderbookLevels*4:2,1].set(1) \
        .at[:,4].set(initid) \
        .at[:,5].set(initid-jnp.arange(0,orderbookLevels*2)) \
        .at[:,6].set(time[0]) \
        .at[:,7].set(time[1])
    return initOB

def get_initial_time(messageData,idx_window):
    return messageData[idx_window,0,0,-2:]


def get_data_messages(messageData,idx_window,step_counter):
    messages=messageData[idx_window,step_counter,:,:]
    return messages
    




# ===================================== #
# ******* Config your own func ******** #
# ===================================== #


@partial(jax.jit)
def get_best_bid(asks, bids):
    L2_state = get_L2_state(1, asks, bids)
    return L2_state[0]

@partial(jax.jit)
def get_best_bid(asks, bids):
    L2_state = get_L2_state(1, asks, bids)
    return L2_state[1]