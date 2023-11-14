"""
JAX Order Book Functionality

University of Oxford
Corresponding Author: Sascha Frey (sascha.frey@st-hughs.ox.ac.uk)
V1.0


Module containing all functions to manipulate the orderbook.
To follow the functional programming paradigm of JAX, the functions of
 the orderbook are not put into an object but are left standalone.

The functions exported are...
add_order: Adds an order to book side.
cancel_order: Removes quantity from book side.
match_order: Match incoming order against standing order.
doNothing: returns unchanged book.
bid_lim: processes a bid limit order.
bid_cancel: processes a bid cancel order.
ask_lim: processes an ask limit order.
ask_cancel: processes an ask cancel order.
cond_type_side: branching depending on order type/side.
cond_type_side_save_states: branching depending on order type/side
                            returns the book&trades for saving in 
                            the jax.scan loop.
cond_type_side_save_bidask: branching depending on order type/side
                            returns the best bid/ask for saving in 
                            the jax.scan loop.
vcond_type_side: jax.vmapped version of cond_type_side
scan_through_entire_array: for loop over all orders to process
scan_through_entire_array_save_states: as above but saving states
scan_through_entire_array_save_bidask: as above but saving bid/ask
vscan_through_entire_array:jax.vmapped version of scan_through_entire_array

_removeZeroQuant: removes order when q<0
_match_bid_order: wrapper of match order
_match_ask_order: wrapper of match order
_get_top_bid_order_idx: return the index in the side for the best bid
_get_top_ask_order_idx: return the index in the side for the best ask
_check_before_matching_bid: conditional for matching while loop on bid
_check_before_matching_ask: conditional for matching while loop on ask
_match_against_bid_orders: match incoming order against bid orders
_match_against_ask_orders: match incoming order against ask orders
"""

from textwrap import fill
from typing import Optional, OrderedDict
from functools import partial, partialmethod

from jax import numpy as jnp
import jax
import chex

INITID = -900000
MAX_INT = 2_147_483_647  # max 32 bit int
#TODO: Get rid of these magic numbers by allowing a config dict to be passed through as a static arg 


############### ADD AND REMOVE ###############
@jax.jit
def add_order(orderside: chex.Array, msg: dict) -> chex.Array :
    """Adds an order to a given side of the orderbook. 
    Will not add negative quantities to the book. 
        Parameters:
                orderside (Array): Array representing bid or ask side
                msg (dict): Message with data of order to add

        Returns:
                orderside (Array): Side of orderbook with added order 
    """
    emptyidx=jnp.where(orderside==-1,size=1,fill_value=-1)[0]
    orderside=orderside.at[emptyidx,:]\
                        .set(jnp.array([
                            msg['price'],
                            jnp.maximum(0,msg['quantity']),
                            msg['orderid'],
                            msg['traderid'],
                            msg['time'],
                            msg['time_ns']]))\
                        .astype(jnp.int32)
    return _removeZeroNegQuant(orderside)

@jax.jit
def _removeZeroNegQuant(orderside):
    """Remove any orders where quant is leq to 0"""
    return jnp.where((orderside[:,1]<=0).reshape((orderside.shape[0],1)),
                        x=(jnp.ones(orderside.shape)*-1).astype(jnp.int32),
                        y=orderside)


@jax.jit
def cancel_order(orderside, msg):
    """Removes quantity of an order from a given side of the orderbook.
    If the resulting order has a remaining quantity of 0 or less it is
    removed entirely. 
    Identifies orders to cancel based on a matching order ID. If there
    is no matching ID, an initial order may also be cancelled provided 
    the price matches. 

        Parameters:
                orderside (Array): Array representing bid or ask side
                msg (dict): Message identifying order to cancel

        Returns:
                orderside (Array): Orderbook side with cancelled order
    """
    def get_init_id_match(orderside, msg):
        """Function to check match of an initial message. Used only 
        if the order ID of the message does not match with an 
        existing order. 
        """
        init_id_match = ((orderside[:, 0] == msg['price']) 
                            & (orderside[:, 2] <= INITID))
        idx = jnp.where(init_id_match, size=1, fill_value=-1)[0][0]
        return idx
    oid_match = (orderside[:, 2] == msg['orderid'])
    idx = jnp.where(oid_match, size=1, fill_value=-1)[0][0]
    idx = jax.lax.cond(idx == -1,
                        get_init_id_match,
                        lambda a, b: idx,
                        orderside,
                        msg)
    orderside = orderside.at[idx, 1].set(orderside[idx, 1] - msg['quantity'])
    return _removeZeroNegQuant(orderside)

############### MATCHING FUNCTIONS ###############


@jax.jit
def match_order(data_tuple):
    """Matches an incoming order against the best order from a given
    side of the order book, and removes the matched quanitity from
    the book. Registers a trade for the size of the matched quantity.
    Returns both the new order book side, as well as the remaining 
    quantity to match.

        Parameters:
                data_tuple (Tuple): 
                    top_order_idx (Int): location of best order in book
                    orderside (Array): Array representing bid/ask side
                    qtm (Int): Quantity of incoming order unmatched
                    price (Int): price of the incoming order
                    trade (Array): Dummy Array representing empty trade
                    agrOID (Int): Order ID of the incoming order
                    time (Int): Arrival time (s) of incoming order
                    time_ns (Int): Arrival time (ns) of incoming order
        
        Returns:
                data_tuple (Tuple): Same as input tuple, but without
                    the top order index and with an updated quantity
                    to match, an updated book side and the resulting 
                    trade. 

    """
    (top_order_idx, orderside, qtm, price,
            trade, agrOID, time, time_ns) = data_tuple
    newquant=jnp.maximum(0,orderside[top_order_idx,1]-qtm) #Could theoretically be removed as an operation because the removeZeroQuand func also removes negatives. 
    qtm=qtm-orderside[top_order_idx,1]
    qtm=qtm.astype(jnp.int32)
    emptyidx=jnp.where(trade==-1,size=1,fill_value=-1)[0]
    trade=trade.at[emptyidx,:] \
                .set(jnp.array([orderside[top_order_idx,0],
                                orderside[top_order_idx,1]-newquant,
                                orderside[top_order_idx,2],
                                [agrOID],
                                [time],
                                [time_ns]]).transpose())
    orderside=_removeZeroNegQuant(orderside.at[top_order_idx,1].set(newquant))
    return (orderside.astype(jnp.int32), jnp.squeeze(qtm),
             price, trade, agrOID, time, time_ns)


@jax.jit
def _match_bid_order(data_tuple):
    """Wrapper to call the matching function and return the index of
      the next best bid order.
    """
    return _get_top_bid_order_idx(data_tuple[1]), *match_order(data_tuple)

@jax.jit
def _match_ask_order(data_tuple):
    """Wrapper to call the matching function and return the index of
      the next best ask order.
    """
    return _get_top_ask_order_idx(data_tuple[1]), *match_order(data_tuple)

@jax.jit
def _get_top_bid_order_idx(orderside):
    """Identifies the index in the array representing the bid side
    which contains the best bid order. This is the order with the
    largest price, with the arrival time acting as the tie-breaker.
    """
    maxPrice=jnp.max(orderside[:,0],axis=0)
    times=jnp.where(orderside[:,0]==maxPrice,orderside[:,4],MAX_INT)
    minTime_s=jnp.min(times,axis=0)
    times_ns=jnp.where(times==minTime_s,orderside[:,5],MAX_INT)
    minTime_ns=jnp.min(times_ns,axis=0)
    return jnp.where(times_ns==minTime_ns,size=1,fill_value=-1)[0]


@jax.jit
def _get_top_ask_order_idx(orderside):
    """Identifies the index in the array representing the ask side
    which contains the best ask order. This is the order with the
    smallest price, with the arrival time acting as the tie-breaker.
    """
    prices=orderside[:,0]
    prices=jnp.where(prices==-1,MAX_INT,prices)
    minPrice=jnp.min(prices)
    times=jnp.where(orderside[:,0]==minPrice,orderside[:,4],MAX_INT)
    minTime_s=jnp.min(times,axis=0)
    times_ns=jnp.where(times==minTime_s,orderside[:,5],MAX_INT)
    minTime_ns=jnp.min(times_ns,axis=0)
    return jnp.where(times_ns==minTime_ns,size=1,fill_value=-1)[0]

@jax.jit
def _check_before_matching_bid(data_tuple):
    """Conditional statement used by the while loop in 
    _match_against_bid_orders which checks if the price of the best bid
    order overlaps the incoming ask order, if there is still unmatched
    quantity in the incoming ask order, and whether there are still bid
    orders in the book. 
    """
    top_order_idx,orderside,qtm,price,trade,_,_,_=data_tuple
    returnarray=((orderside[top_order_idx,0]>=price)
                  & (qtm>0)
                  & (orderside[top_order_idx,0]!=-1))
    return jnp.squeeze(returnarray)

@jax.jit
def _match_against_bid_orders(orderside,qtm,price,trade,agrOID,time,time_ns):
    """Wrapper for the while loop that gets the top bid order, and
    matches the incoming order against it whilst the 
    _check_before_matching_bid function remains true.
    Returns the new set of bid orders after matching, and the remaining
    quantity to match/
    """
    top_order_idx=_get_top_bid_order_idx(orderside)
    (top_order_idx,orderside,
     qtm,price,trade,_,_,_)=jax.lax.while_loop(_check_before_matching_bid,
                                               _match_bid_order,
                                               (top_order_idx,orderside,
                                                qtm,price,trade,agrOID,
                                                time,time_ns))
    return (orderside,qtm,price,trade)

@jax.jit
def _check_before_matching_ask(data_tuple):
    """Conditional statement used by the while loop in 
    _match_against_ask_orders which checks if the price of the best ask
    order overlaps the incoming ask order, if there is still unmatched
    quantity in the incoming ask order, and whether there are still bid
    orders in the book. 
    """
    top_order_idx,orderside,qtm,price,trade,_,_,_=data_tuple
    returnarray=((orderside[top_order_idx,0]<=price)
                  & (qtm>0) 
                  & (orderside[top_order_idx,0]!=-1))
    return jnp.squeeze(returnarray)

@jax.jit
def _match_against_ask_orders(orderside,qtm,price,trade,agrOID,time,time_ns):
    """Wrapper for the while loop that gets the top ask order, and
    matches the incoming order against it whilst the 
    _check_before_matching_ask function remains true.
    Returns the new set of bid orders after matching, and the remaining
    quantity to match.
    """
    top_order_idx=_get_top_ask_order_idx(orderside)
    (top_order_idx,orderside,
     qtm,price,trade,_,_,_)=jax.lax.while_loop(_check_before_matching_ask,
                                               _match_ask_order,
                                               (top_order_idx,orderside,
                                                qtm,price,trade,agrOID,
                                                time,time_ns))
    return (orderside,qtm,price,trade)

######## TYPE AND SIDE FUNCTIONS #############

def doNothing(msg,askside,bidside,trades):
    """Dummy function for conditional statements to do nothing
    in certain cases. 

        Parameters:
                msg (Dict): Incoming message to process.
                    quantity (Int): Quantity to buy/sell
                    price (Int): Price of order
                    orderid (Int): Unique ID in the book
                    traderid (Int): Trader ID, rarely available
                    time (Int): Time of arrival (full seconds)
                    time_ns (Int): Time of arrival (remaining ns)
                askside (Array): All ask orders in book
                bidside (Array): All bid orders in book
                trades (Array): Running count of all occured trades
                
        Returns:
                askside (Array): Same as parameter, after processing
                bidside (Array): Same as parameter, after processing
                trades (Array): Same as parameter, after processing
    """
    return askside,bidside,trades

def bid_lim(msg,askside,bidside,trades):
    """Function for processing a limit order to bid. After attempting
    to match with the ask side, the remaining quantity of the order is
    added to the bid side of the limit order book.

        Parameters:
                msg (Dict): Incoming message to process.
                    quantity (Int): Quantity to buy/sell
                    price (Int): Price of order
                    orderid (Int): Unique ID in the book
                    traderid (Int): Trader ID, rarely available
                    time (Int): Time of arrival (full seconds)
                    time_ns (Int): Time of arrival (remaining ns)
                askside (Array): All ask orders in book
                bidside (Array): All bid orders in book
                trades (Array): Running count of all occured trades
                
        Returns:
                askside (Array): Same as parameter, after processing
                bidside (Array): Same as parameter, after processing
                trades (Array): Same as parameter, after processing
    """
    matchtuple=_match_against_ask_orders(askside,msg["quantity"],
                                         msg["price"],
                                         trades,
                                         msg['orderid'],
                                         msg["time"],
                                         msg["time_ns"])
    msg["quantity"]=matchtuple[1] #Remaining quantity
    bids=add_order(bidside,msg)
    return matchtuple[0],bids,matchtuple[3]

def bid_cancel(msg,askside,bidside,trades):
    """Function for processing a cancel order on the bid side.
    Simply calls the cancel operation on the bid side. 

        Parameters:
                msg (Dict): Incoming message to process.
                    quantity (Int): Quantity to buy/sell
                    price (Int): Price of order
                    orderid (Int): Unique ID in the book
                    traderid (Int): Trader ID, rarely available
                    time (Int): Time of arrival (full seconds)
                    time_ns (Int): Time of arrival (remaining ns)
                askside (Array): All ask orders in book
                bidside (Array): All bid orders in book
                trades (Array): Running count of all occured trades
                
        Returns:
                askside (Array): Same as parameter, after processing
                bidside (Array): Same as parameter, after processing
                trades (Array): Same as parameter, after processing
    """
    return askside,cancel_order(bidside,msg),trades


def ask_lim(msg,askside,bidside,trades):
    """Function for processing a limit order to ask. After attempting
    to match with the bid side, the remaining quantity of the order is
    added to the ask side of the limit order book.

        Parameters:
                msg (Dict): Incoming message to process.
                    quantity (Int): Quantity to buy/sell
                    price (Int): Price of order
                    orderid (Int): Unique ID in the book
                    traderid (Int): Trader ID, rarely available
                    time (Int): Time of arrival (full seconds)
                    time_ns (Int): Time of arrival (remaining ns)
                askside (Array): All ask orders in book
                bidside (Array): All bid orders in book
                trades (Array): Running count of all occured trades
                
        Returns:
                askside (Array): Same as parameter, after processing
                bidside (Array): Same as parameter, after processing
                trades (Array): Same as parameter, after processing
    """
    matchtuple=_match_against_bid_orders(bidside,
                                         msg["quantity"],
                                         msg["price"],
                                         trades,
                                         msg['orderid'],
                                         msg["time"],
                                         msg["time_ns"])
    msg["quantity"]=matchtuple[1] #Remaining quantity
    asks=add_order(askside,msg)
    return asks,matchtuple[0],matchtuple[3]

def ask_cancel(msg,askside,bidside,trades):
    """Function for processing a cancel order on the ask side.
    Simply calls the cancel operation on the ask side. 

        Parameters:
                msg (Dict): Incoming message to process.
                    quantity (Int): Quantity to buy/sell
                    price (Int): Price of order
                    orderid (Int): Unique ID in the book
                    traderid (Int): Trader ID, rarely available
                    time (Int): Time of arrival (full seconds)
                    time_ns (Int): Time of arrival (remaining ns)
                askside (Array): All ask orders in book
                bidside (Array): All bid orders in book
                trades (Array): Running count of all occured trades
                
        Returns:
                askside (Array): Same as parameter, after processing
                bidside (Array): Same as parameter, after processing
                trades (Array): Same as parameter, after processing
    """
    return cancel_order(askside,msg),bidside,trades


###############  BRANCHING FUNCTIONS ###############
@jax.jit
def cond_type_side(book_state, data):
    """Branching function which calls the relevant function based on
    the side and type fields of the incoming message. Organises the 
    array from data as a message Dict. 

        Parameters:
                book_state (Tuple): State of the orderbook
                    askside (Array): All ask orders in the book
                    bidside (Array): All bid orders in the book
                    trades (Array): All trades which have occured
                data (Array): Vector containing message content
                
        Returns:
                book_state (Tuple): Same as parameter after processing
                    the message in data
                book_state_to_save (Int): 0, nothing saved in lax.scan
    """
    askside,bidside,trades=book_state
    msg={'side':data[1],
         'type':data[0],
         'price':data[3],
         'quantity':data[2],
         'orderid':data[5],
         'traderid':data[4],
         'time':data[6],
         'time_ns':data[7]}
    
    s = msg["side"]
    t = msg["type"]
    index = ((((s == -1) & (t == 1)) | ((s ==  1) & (t == 4))) * 0 
             + (((s ==  1) & (t == 1)) | ((s == -1) & (t == 4))) * 1
             + (((s == -1) & (t == 2)) | ((s == -1) & (t == 3))) * 2 
             + (((s ==  1) & (t == 2)) | ((s ==  1) & (t == 3))) * 3)

    ask, bid, trade = jax.lax.switch(index,
                                     (ask_lim, bid_lim,
                                       ask_cancel, bid_cancel),
                                     msg,
                                     askside,
                                     bidside,
                                     trades)
    return (ask, bid, trade), 0

@jax.jit
def cond_type_side_save_states(ordersides,data):
    """Branching function which calls the relevant function based on
    the side and type fields of the incoming message. Organises the 
    array from data as a message Dict. Addtionally, returns the order 
    book state tuple to be saved by the lax.scan function which this
    function is designed to be called by.

        Parameters:
                book_state (Tuple): State of the orderbook
                    askside (Array): All ask orders in the book
                    bidside (Array): All bid orders in the book
                    trades (Array): All trades which have occured
                data (Array): Vector containing message content
                
        Returns:
                book_state (Tuple): Same as parameter after processing
                    the message in data
                book_state_to_save (Int): book_state
    """
    askside,bidside,trades=ordersides
    msg={'side':data[1],
         'type':data[0],
         'price':data[3],
         'quantity':data[2],
         'orderid':data[5],
         'traderid':data[4],
         'time':data[6],
         'time_ns':data[7]}

    s = msg["side"]
    t = msg["type"]
    index = ((((s == -1) & (t == 1)) | ((s ==  1) & (t == 4))) * 0
             + (((s ==  1) & (t == 1)) | ((s == -1) & (t == 4))) * 1 
             + (((s == -1) & (t == 2)) | ((s == -1) & (t == 3))) * 2
             + (((s ==  1) & (t == 2)) | ((s ==  1) & (t == 3))) * 3)
    ask, bid, trade = jax.lax.switch(index, (ask_lim, bid_lim, ask_cancel, bid_cancel), msg, askside, bidside, trades)
    return (ask,bid,trade),(ask,bid,trade)

@jax.jit
def cond_type_side_save_bidask(ordersides,data):
    """Branching function which calls the relevant function based on
    the side and type fields of the incoming message. Organises the 
    array from data as a message Dict. Addtionally, returns the order 
    book state tuple to be saved by the lax.scan function which this
    function is designed to be called by.

        Parameters:
                book_state (Tuple): State of the orderbook
                    askside (Array): All ask orders in the book
                    bidside (Array): All bid orders in the book
                    trades (Array): All trades which have occured
                data (Array): Vector containing message content
                
        Returns:
                book_state (Tuple): Same as parameter after processing
                    the message in data
                book_state_to_save (Int): best bid/ask price & quant
    """
    askside,bidside,trades=ordersides
    msg={'side':data[1],
         'type':data[0],
         'price':data[3],
         'quantity':data[2],
         'orderid':data[5],
         'traderid':data[4],
         'time':data[6],
         'time_ns':data[7]}

    s = msg["side"]
    t = msg["type"]
    index = ((((s == -1) & (t == 1)) | ((s ==  1) & (t == 4))) * 0
             + (((s ==  1) & (t == 1)) | ((s == -1) & (t == 4))) * 1 
             + (((s == -1) & (t == 2)) | ((s == -1) & (t == 3))) * 2
             + (((s ==  1) & (t == 2)) | ((s ==  1) & (t == 3))) * 3)
    ask, bid, trade = jax.lax.switch(index, (ask_lim, bid_lim, ask_cancel, bid_cancel), msg, askside, bidside, trades)
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
def get_size(bookside,agentID):
    """ """
    return jnp.sum(jnp.where(bookside[:,3]==agentID,1,0)).astype(jnp.int32)

def getCancelMsgs(bookside,agentID,size,side):
    """Obtain messages to cancel based on a given ID to lookup.
        Currently only used in the execution environment."""
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
"""def getCancelMsgs_smart(bookside,agentID,size,side,action_msgs):
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
    return cancel_msgs"""


   





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
        idx = _get_top_ask_order_idx(side_array)
    # best buy order / bid
    elif side == 1:
        idx = _get_top_bid_order_idx(side_array)
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
