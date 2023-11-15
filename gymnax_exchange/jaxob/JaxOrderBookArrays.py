"""Module containing all functions to manipulate the orderbook.

To follow the functional programming paradigm of JAX, the functions of the
orderbook are not put into an object but are left standalone.

The functions exported are:

add_order : Adds a given order to the given orderside 
cancel_order : Removes quantity (and order if remainder <0) from a given order side
match_order : remove quantity from a single standing order and generate trades.



"""


from textwrap import fill
from typing import Optional, OrderedDict
from jax import numpy as jnp
import jax
from functools import partial, partialmethod
import chex

INITID = -9000
#MAXPRICE=999999999
#TODO: Get rid of these magic numbers by allowing a config dict to be passed through as a static arg 

MAX_INT = 2_147_483_647  # max 32 bit int

############### ADD AND REMOVE ###############
@jax.jit
def add_order(orderside: chex.Array, msg: dict) -> chex.Array :
    """Low level function that will add an order (Dict)
      to the orderbook (Array) and return the updated"""
    emptyidx=jnp.where(orderside==-1,size=1,fill_value=-1)[0]
    orderside=orderside.at[emptyidx,:].set(jnp.array([msg['price'],jnp.maximum(0,msg['quantity']),msg['orderid'],msg['traderid'],msg['time'],msg['time_ns']])).astype(jnp.int32)
    return __removeZeroNegQuant(orderside)

@jax.jit
def __removeZeroNegQuant(orderside):
    return jnp.where((orderside[:,1]<=0).reshape((orderside.shape[0],1)),x=(jnp.ones(orderside.shape)*-1).astype(jnp.int32),y=orderside)


# @jax.jit
# def cancel_order(orderside,msg):
#     # jax.debug.breakpoint()
#     condition=((orderside[:,2]==msg['orderid']) | ((orderside[:,0]==msg['price']) & (orderside[:,2]<=-9000)))
#     idx=jnp.where(condition,size=1,fill_value=-1)[0]
#     orderside=orderside.at[idx,1].set(orderside[idx,1]-msg['quantity'])
#     return __removeZeroNegQuant(orderside)

@jax.jit
def cancel_order(orderside, msg):
    def get_init_id_match(orderside, msg):
        init_id_match = ((orderside[:, 0] == msg['price']) & (orderside[:, 2] <= INITID))
        idx = jnp.where(init_id_match, size=1, fill_value=-1)[0][0]
        return idx

    # jax.debug.breakpoint()
    # TODO: also check for price here?
    oid_match = (orderside[:, 2] == msg['orderid'])
    idx = jnp.where(oid_match, size=1, fill_value=-1)[0][0]
    idx = jax.lax.cond(idx == -1, get_init_id_match, lambda a, b: idx, orderside, msg)
    orderside = orderside.at[idx, 1].set(orderside[idx, 1] - msg['quantity'])
    return __removeZeroNegQuant(orderside)

############### MATCHING FUNCTIONS ###############

@jax.jit
def match_bid_order(data_tuple):
    matching_tuple = match_order(data_tuple)
    top_i = __get_top_bid_order_idx(matching_tuple[0])
    return top_i, *matching_tuple

@jax.jit
def match_ask_order(data_tuple):
    matching_tuple = match_order(data_tuple)
    top_i = __get_top_ask_order_idx(matching_tuple[0])
    return top_i, *matching_tuple

@jax.jit
def match_order(data_tuple):
    top_order_idx, orderside, qtm, price, trade, agrOID, time, time_ns = data_tuple
    newquant=jnp.maximum(0,orderside[top_order_idx,1]-qtm) #Could theoretically be removed as an operation because the removeZeroQuand func also removes negatives. 
    qtm=qtm-orderside[top_order_idx,1]
    qtm=qtm.astype(jnp.int32)
    emptyidx=jnp.where(trade==-1,size=1,fill_value=-1)[0]
    trade=trade.at[emptyidx,:].set(jnp.array([orderside[top_order_idx,0],orderside[top_order_idx,1]-newquant,orderside[top_order_idx,2],[agrOID],[time],[time_ns]]).transpose())
    orderside=__removeZeroNegQuant(orderside.at[top_order_idx,1].set(newquant))
    return (orderside.astype(jnp.int32), jnp.squeeze(qtm), price, trade, agrOID, time, time_ns)

@jax.jit
def __get_top_bid_order_idx(orderside):
    maxPrice=jnp.max(orderside[:,0],axis=0)
    times=jnp.where(orderside[:,0]==maxPrice,orderside[:,4],999999999)
    minTime_s=jnp.min(times,axis=0)
    times_ns=jnp.where(times==minTime_s,orderside[:,5],999999999)
    minTime_ns=jnp.min(times_ns,axis=0)
    return jnp.where(times_ns==minTime_ns,size=1,fill_value=-1)[0]

@jax.jit
def __get_top_ask_order_idx(orderside):
    prices=orderside[:,0]
    prices=jnp.where(prices==-1, 999999999, prices)
    minPrice=jnp.min(prices)
    times=jnp.where(orderside[:,0]==minPrice, orderside[:,4], 999999999)
    minTime_s=jnp.min(times,axis=0)
    times_ns=jnp.where(times==minTime_s,orderside[:,5],999999999)
    minTime_ns=jnp.min(times_ns,axis=0)
    return jnp.where(times_ns==minTime_ns,size=1,fill_value=-1)[0]

@jax.jit
def __check_before_matching_bid(data_tuple):
    top_order_idx,orderside,qtm,price,trade,_,_,_=data_tuple
    returnarray=(orderside[top_order_idx,0]>=price) & (qtm>0) & (orderside[top_order_idx,0]!=-1)
    return jnp.squeeze(returnarray)

@jax.jit
def _match_against_bid_orders(orderside,qtm,price,trade,agrOID,time,time_ns):
    top_order_idx=__get_top_bid_order_idx(orderside)
    top_order_idx,orderside,qtm,price,trade,_,_,_=jax.lax.while_loop(__check_before_matching_bid,match_bid_order,(top_order_idx,orderside,qtm,price,trade,agrOID,time,time_ns))
    return (orderside,qtm,price,trade)

@jax.jit
def __check_before_matching_ask(data_tuple):
    # jax.debug.print('data_tuple: {}', data_tuple)
    top_order_idx,orderside,qtm,price,trade,_,_,_=data_tuple
    returnarray=(orderside[top_order_idx,0]<=price) & (qtm>0) & (orderside[top_order_idx,0]!=-1)
    return jnp.squeeze(returnarray)

@jax.jit
def _match_against_ask_orders(orderside,qtm,price,trade,agrOID,time,time_ns):
    top_order_idx=__get_top_ask_order_idx(orderside)
    top_order_idx,orderside,qtm,price,trade,_,_,_=jax.lax.while_loop(__check_before_matching_ask,match_ask_order,(top_order_idx,orderside,qtm,price,trade,agrOID,time,time_ns))
    return (orderside,qtm,price,trade)

########Type Functions#############

def doNothing(msg,askside,bidside,trades):
    return askside,bidside,trades

def bid_lim(msg,askside,bidside,trades):
    #match with asks side
    #add remainder to bids side
    matchtuple=_match_against_ask_orders(askside,msg["quantity"],msg["price"],trades,msg['orderid'],msg["time"],msg["time_ns"])
    #^(orderside,qtm,price,trade)
    msg["quantity"]=matchtuple[1]
    bids=add_order(bidside,msg)
    return matchtuple[0],bids,matchtuple[3]

def bid_cancel(msg,askside,bidside,trades):
    return askside,cancel_order(bidside,msg),trades

def bid_mkt(msg,askside,bidside,trades):
    msg["price"]=999999999
    matchtuple=_match_against_ask_orders(askside,msg["quantity"],msg["price"],trades,msg['orderid'],msg["time"],msg["time_ns"])
    #^(orderside,qtm,price,trade)
    return matchtuple[0],bidside,matchtuple[3]


def ask_lim(msg,askside,bidside,trades):
    #match with bids side
    #add remainder to asks side
    matchtuple=_match_against_bid_orders(bidside,msg["quantity"],msg["price"],trades,msg['orderid'],msg["time"],msg["time_ns"])
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
    matchtuple=_match_against_bid_orders(bidside,msg["quantity"],msg["price"],trades,msg['orderid'],msg["time"],msg["time_ns"])
    #^(orderside,qtm,price,trade)
    return askside,matchtuple[0],matchtuple[3]


############### MAIN BRANCHING FUNCS ###############

@jax.jit
def cond_type_side(book_state, data):
    askside,bidside,trades=book_state
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
    # index=((msg["side"]+1)+msg["type"]).astype(jnp.int32)
    s = msg["side"]
    t = msg["type"]
    index = (((s == -1) & (t == 1)) | ((s ==  1) & (t == 4))) * 0 + \
            (((s ==  1) & (t == 1)) | ((s == -1) & (t == 4))) * 1 + \
            (((s == -1) & (t == 2)) | ((s == -1) & (t == 3))) * 2 + \
            (((s ==  1) & (t == 2)) | ((s ==  1) & (t == 3))) * 3
    # jax.debug.print("msg[side] {}", msg["side"])
    # jax.debug.print("msg[type] {}", msg["type"])
    # jax.debug.print("index is {}", index)
    # ask,bid,trade=jax.lax.switch(index-1,(ask_lim,ask_cancel,bid_lim,bid_cancel),msg,askside,bidside,trades)
    ask, bid, trade = jax.lax.switch(index, (ask_lim, bid_lim, ask_cancel, bid_cancel), msg, askside, bidside, trades)
    return (ask, bid, trade), 0

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
    # index=((msg["side"]+1)+msg["type"]).astype(jnp.int32)
    s = msg["side"]
    t = msg["type"]
    index = (((s == -1) & (t == 1)) | ((s ==  1) & (t == 4))) * 0 + \
            (((s ==  1) & (t == 1)) | ((s == -1) & (t == 4))) * 1 + \
            (((s == -1) & (t == 2)) | ((s == -1) & (t == 3))) * 2 + \
            (((s ==  1) & (t == 2)) | ((s ==  1) & (t == 3))) * 3
    # ask,bid,trade=jax.lax.switch(index-1,(ask_lim,ask_cancel,bid_lim,bid_cancel),msg,askside,bidside,trades)
    ask, bid, trade = jax.lax.switch(index, (ask_lim, bid_lim, ask_cancel, bid_cancel), msg, askside, bidside, trades)
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
    # index=((msg["side"]+1)+msg["type"]).astype(jnp.int32)
    s = msg["side"]
    t = msg["type"]
    index = (((s == -1) & (t == 1)) | ((s ==  1) & (t == 4))) * 0 + \
            (((s ==  1) & (t == 1)) | ((s == -1) & (t == 4))) * 1 + \
            (((s == -1) & (t == 2)) | ((s == -1) & (t == 3))) * 2 + \
            (((s ==  1) & (t == 2)) | ((s ==  1) & (t == 3))) * 3
    # ask,bid,trade=jax.lax.switch(index-1,(ask_lim,ask_cancel,bid_lim,bid_cancel),msg,askside,bidside,trades)
    ask, bid, trade = jax.lax.switch(index, (ask_lim, bid_lim, ask_cancel, bid_cancel), msg, askside, bidside, trades)
    
    l2_state = get_L2_state(ask, bid, 5)
    b_ask, b_bid = get_best_bid_and_ask(ask, bid)
    # jax.debug.print('l2_state\n{}', l2_state)
    # jax.debug.print('{} ask-bid', b_ask - b_bid)
    # jax.debug.breakpoint()
    
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
    # jax.debug.print('before scan_through_entire_array_save_bidask')
    # jax.debug.breakpoint()
    last,all=jax.lax.scan(cond_type_side_save_bidask,ordersides,msg_array)
    # jax.debug.print('after scan_through_entire_array_save_bidask')
    # jax.debug.breakpoint()
    return (last[0],last[1],last[2],all[0][-steplines:],all[1][-steplines:])

vscan_through_entire_array=jax.vmap(scan_through_entire_array,(2,(0,0,0)),0)

################ GET CANCEL MESSAGES ################

#Obtain messages to cancel based on a given ID to lookup. Currently only used in the execution environment.
def get_size(bookside,agentID):
    return jnp.sum(jnp.where(bookside[:,3]==agentID,1,0)).astype(jnp.int32)

def getCancelMsgs(bookside,agentID,size,side):
    bookside=jnp.concatenate([bookside,jnp.zeros((1,6),dtype=jnp.int32)],axis=0)
    indices_to_cancel=jnp.where(bookside[:,3]==agentID,size=size,fill_value=-1)
    cancel_msgs=jnp.concatenate([jnp.ones((1,size),dtype=jnp.int32)*2,
                                 jnp.ones((1,size),dtype=jnp.int32)*side,
                                bookside[indices_to_cancel,1],
                                bookside[indices_to_cancel,0],
                                bookside[indices_to_cancel,3],
                                bookside[indices_to_cancel,2],
                                bookside[indices_to_cancel,4],
                                bookside[indices_to_cancel,5]],axis=0).transpose()
    return cancel_msgs


#TODO Currently not implemented: should be used for a less naive version of the autocancel
def getCancelMsgs_smart(bookside,agentID,size,side,action_msgs):
    cond=jnp.stack([bookside[:,3]==agentID]*6,axis=1)
    indices_to_cancel=jnp.where(bookside[:,3]==agentID,size=size,fill_value=0)
    cancel_msgs=jnp.concatenate([jnp.ones((1,size),dtype=jnp.int32)*2,
                                jnp.ones((1,size),dtype=jnp.int32)*side,
                                bookside[indices_to_cancel,1],
                                bookside[indices_to_cancel,0],
                                bookside[indices_to_cancel,3],
                                bookside[indices_to_cancel,2],
                                bookside[indices_to_cancel,4],
                                bookside[indices_to_cancel,5]],axis=0).transpose()
    cancel_msgs=jnp.where(cancel_msgs==-1,0,cancel_msgs)
    jax.lax.scan(remove_cnl_if_renewed,cancel_msgs,action_msgs)
    return cancel_msgs

#TODO Currently not implemented: should be used for a less naive version of the autocancel
def remove_cnl_if_renewed(cancel_msgs,action_msg):
    jnp.where(cancel_msgs[:,3]==action_msg[3],)
    return cancel_msgs

   





######Helper functions for getting information #######


def get_totquant_at_price(orderside,price):
        return jnp.sum(jnp.where(orderside[:,0]==price,orderside[:,1],0))

get_totquant_at_prices=jax.vmap(get_totquant_at_price,(None,0),0)

# @partial(jax.jit,static_argnums=0)
# def get_L2_state(N,asks,bids):
#     bid_prices=-jnp.unique(-bids[:,0],size=N,fill_value=1)
#     ask_prices=jnp.unique(jnp.where(asks[:,0]==-1,999999999,asks[:,0]),size=N,fill_value=999999999)
#     ask_prices=jnp.where(ask_prices==999999999,-1,ask_prices)

#     bid_quants=get_totquant_at_prices(bids,bid_prices)
#     ask_quants=get_totquant_at_prices(asks,ask_prices)
#     bid_quants=jnp.where(bid_quants<0,0,bid_quants)
#     ask_quants=jnp.where(ask_quants<0,0,ask_quants)
#     return jnp.stack((ask_prices,ask_quants,bid_prices,bid_quants),axis=1,dtype=jnp.int32)

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

@jax.jit
def init_msgs_from_l2(
    book_l2: jnp.ndarray,
    time: Optional[jax.Array] = None,
) -> jax.Array:
    """  """
    orderbookLevels = book_l2.shape[0] // 4  # price/quantity for bid/ask
    data = book_l2.reshape(orderbookLevels * 2, 2)
    newarr = jnp.zeros((orderbookLevels * 2, 8), dtype=jnp.int32)
    if time is None:
        time = jnp.array([34200, 0])
    initOB = newarr \
        .at[:, 3].set(data[:,0]) \
        .at[:, 2].set(data[:,1]) \
        .at[:, 0].set(1) \
        .at[0:orderbookLevels*4:2, 1].set(-1) \
        .at[1:orderbookLevels*4:2, 1].set(1) \
        .at[:, 4].set(INITID) \
        .at[:, 5].set(INITID - jnp.arange(0, orderbookLevels*2)) \
        .at[:, 6].set(time[0]) \
        .at[:, 7].set(time[1])
    return initOB


# #TODO: Actually complete this function to not only return dummy vars
# def get_initial_orders(bookData,idx_window,time):
#     orderbookLevels=10
#     initid=-9000
#     data=jnp.array(bookData[idx_window]).reshape(int(10*2),2)
#     newarr = jnp.zeros((int(orderbookLevels*2),8),dtype=jnp.int32)
#     initOB = newarr \
#         .at[:,3].set(data[:,0]) \
#         .at[:,2].set(data[:,1]) \
#         .at[:,0].set(1) \
#         .at[0:orderbookLevels*4:2,1].set(-1) \
#         .at[1:orderbookLevels*4:2,1].set(1) \
#         .at[:,4].set(initid) \
#         .at[:,5].set(initid-jnp.arange(0,orderbookLevels*2)) \
#         .at[:,6].set(time[0]) \
#         .at[:,7].set(time[1])
#     return initOB

def get_initial_time(messageData,idx_window):
    return messageData[idx_window,0,0,-2:]


def get_data_messages(messageData,idx_window,step_counter):
    messages=messageData[idx_window,step_counter,:,:]
    return messages
    




# ===================================== #
# ******* Config your own func ******** #
# ===================================== #


@jax.jit
def get_best_ask(asks):
    '''  Return the best / lowest ask price. If there is no ask, return -1. '''
    min = jnp.min(jnp.where(asks[:, 0] == -1, MAX_INT, asks[:, 0]))
    return jnp.where(min == MAX_INT, -1, min)

@jax.jit
def get_best_bid(bids):
    ''' Return the best / highest bid price. If there is no bid, return -1. '''
    return jnp.max(bids[:, 0])

@jax.jit
def get_volume_at_price(side_array, price):
    volume = jnp.sum(jnp.where(side_array[:,0] == price, side_array[:,1], 0))
    return volume

@jax.jit
def get_init_volume_at_price(
        side_array: jax.Array,
        price: int
    ) -> jax.Array:
    ''' Returns the size of initial volume (order with INITID) at given price. '''
    volume = jnp.sum(
        jnp.where(
            (side_array[:, 0] == price) & (side_array[:, 2] <= INITID), 
            side_array[:, 1], 
            0
        )
    )
    return volume

@jax.jit
def get_order_by_id(
        side_array: jax.Array,
        order_id: int,
    ) -> jax.Array:
    """ Returns all order fields for the first order matching the given order_id.
        CAVE: if the same ID is used multiple times, will only return the first 
        (e.g. for INITID)
    """
    idx = jnp.where( 
        side_array[..., 2] == order_id,
        size=1,
        fill_value=-1,
    )
    # return vector of -1 if not found
    return jax.lax.cond(
        idx == -1,
        lambda i: -1 * jnp.ones((6,), dtype=jnp.int32),
        lambda i: side_array[i][0],
        idx
    )

@jax.jit
def get_order_by_id_and_price(
        side_array: jax.Array,
        order_id: int,
        price: int,
    ) -> jax.Array:
    """ Returns all order fields for the first order matching the given order_id at the given price.
        CAVE: if the same ID is used multiple times at the given price level, will only return the first 
    """
    idx = jnp.where(
        ((side_array[..., 2] == order_id) &
         (side_array[..., 0] == price)),
        size=1,
        fill_value=-1,
    )
    # return vector of -1 if not found
    return jax.lax.cond(
        idx == -1,
        lambda i: -1 * jnp.ones((6,), dtype=jnp.int32),
        lambda i: side_array[i][0],
        idx
    )

@jax.jit
def get_order_ids(
        orderbook_array: jax.Array,
    ) -> jax.Array:
    """ Returns a list of all order ids in the orderbook
    """
    return jnp.unique(orderbook_array[:, 2], size=orderbook_array.shape[0], fill_value=1)

@partial(jax.jit, static_argnums=0)
def get_next_executable_order(side, side_array):
    # best sell order / ask
    if side == 0:
        idx = __get_top_ask_order_idx(side_array)
    # best buy order / bid
    elif side == 1:
        idx = __get_top_bid_order_idx(side_array)
    else:
        raise ValueError("Side must be 0 (bid) or 1 (ask).")
    return side_array[idx].squeeze()

@partial(jax.jit, static_argnums=2)
def get_L2_state(asks, bids, n_levels):
    # unique sorts ascending --> negative values to get descending
    bid_prices = -1 * jnp.unique(-1 * bids[:, 0], size=n_levels, fill_value=1)
    # replace -1 with max 32 bit int in sorting asks before sorting
    ask_prices = jnp.unique(
        jnp.where(asks[:, 0] == -1, MAX_INT, asks[:, 0]),
        size=n_levels,
        fill_value=-1
    )
    # replace max 32 bit int with -1 after sorting
    ask_prices = jnp.where(ask_prices == MAX_INT, -1, ask_prices)

    bids = jnp.stack((bid_prices, get_totquant_at_prices(bids, bid_prices)))
    asks = jnp.stack((ask_prices, get_totquant_at_prices(asks, ask_prices)))
    # set negative volumes to 0
    bids = bids.at[1].set(jnp.where(bids[1] < 0, 0, bids[1]))
    asks = asks.at[1].set(jnp.where(asks[1] < 0, 0, asks[1]))
    # combine asks and bids in joint representation
    l2_state = jnp.hstack((asks.T, bids.T)).flatten()
    return l2_state

vmap_get_L2_state = jax.vmap(get_L2_state, (0, 0, None), 0)
