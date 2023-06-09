from textwrap import fill
from typing import OrderedDict
from jax import numpy as jnp
import jax
from functools import partial, partialmethod

INITID=-9000
MAXPRICE=999999999


@jax.jit
def add_order(orderside,msg):
    emptyidx=jnp.where(orderside==-1,size=1,fill_value=-1)[0]
    return orderside.at[emptyidx,:].set(jnp.array([msg['price'],msg['quantity'],msg['orderid'],msg['traderid'],msg['time'],msg['time_ns']])).astype("int32")

@jax.jit
def removeZeroQuant(orderside):
    return jnp.where((orderside[:,1]<=0).reshape((orderside.shape[0],1)),x=(jnp.ones(orderside.shape)*-1).astype("int32"),y=orderside)


@jax.jit
def cancel_order(orderside,msg):
    condition=((orderside[:,2]==msg['orderid']) | ((orderside[:,0]==msg['price']) & (orderside[:,2]<=INITID)))
    idx=jnp.where(condition,size=1,fill_value=-1)[0]
    orderside=orderside.at[idx,1].set(jnp.minimum(0,orderside[idx,1]-msg['quantity']))
    return removeZeroQuant(orderside)

@jax.jit
def match_order(data_tuple):
    orderside,qtm,price,top_order_idx,trade=data_tuple
    newquant=jnp.maximum(0,orderside[top_order_idx,1]-qtm)
    qtm=qtm-orderside[top_order_idx,1]
    qtm=qtm.astype("int32")
    emptyidx=jnp.where(trade==-1,size=1,fill_value=-1)[0]
    trade=trade.at[emptyidx,:].set(jnp.array([orderside[top_order_idx,0],orderside[top_order_idx,1]-newquant,orderside[top_order_idx,2],orderside[top_order_idx,4],orderside[top_order_idx,5]]).transpose())
    orderside=removeZeroQuant(orderside.at[top_order_idx,1].set(newquant))
    top_order_idx=get_top_bid_order_idx(orderside)
    return (orderside.astype("int32"),jnp.squeeze(qtm),price,top_order_idx,trade)

@jax.jit
def get_top_bid_order_idx(orderside):
    maxPrice=jnp.max(orderside[:,0],axis=0)
    times=jnp.where(orderside[:,0]==maxPrice,orderside[:,4],MAXPRICE)
    minTime_s=jnp.min(times,axis=0)
    times_ns=jnp.where(times==minTime_s,orderside[:,5],MAXPRICE)
    minTime_ns=jnp.min(times_ns,axis=0)
    return jnp.where(times_ns==minTime_ns,size=1,fill_value=-1)[0]

@jax.jit
def check_before_matching_bid(data_tuple):
    orderside,qtm,price,top_order_idx,trade=data_tuple
    returnarray=(orderside[top_order_idx,0]>=price) & (qtm>0) & (orderside[top_order_idx,0]!=-1)
    return jnp.squeeze(returnarray)

@jax.jit
def match_against_bid_orders(orderside,qtm,price,trade):
    top_order_idx=get_top_bid_order_idx(orderside)
    orderside,qtm,price,top_order_idx,trade=jax.lax.while_loop(check_before_matching_bid,match_order,(orderside,qtm,price,top_order_idx,trade))
    return (orderside,qtm,price,trade)

@jax.jit
def check_before_matching_ask(data_tuple):
    orderside,qtm,price,top_order_idx,trade=data_tuple
    returnarray=(orderside[top_order_idx,0]<=price) & (qtm>0) & (orderside[top_order_idx,0]!=-1)
    return jnp.squeeze(returnarray)

@jax.jit
def match_against_ask_orders(orderside,qtm,price,trade):
    top_order_idx=get_top_ask_order_idx(orderside)
    orderside,qtm,price,top_order_idx,trade=jax.lax.while_loop(check_before_matching_ask,match_order,(orderside,qtm,price,top_order_idx,trade))
    return (orderside,qtm,price,trade)

@jax.jit
def get_top_ask_order_idx(orderside):
    prices=orderside[:,0]
    prices=jnp.where(prices==-1,MAXPRICE,prices)
    minPrice=jnp.min(prices)
    times=jnp.where(orderside[:,0]==minPrice,orderside[:,4],MAXPRICE)
    minTime_s=jnp.min(times,axis=0)
    times_ns=jnp.where(times==minTime_s,orderside[:,5],MAXPRICE)
    minTime_ns=jnp.min(times_ns,axis=0)
    return jnp.where(times_ns==minTime_ns,size=1,fill_value=-1)[0]

@jax.jit
def cond_type_side(ordersides,data):
    askside,bidside=ordersides
    msg={
    'side':data[1],
    'type':data[0],
    'price':data[3],
    'quantity':data[2],
    'orderid':data[5],
    'traderid':data[4],
    'time':data[6],
    'time_ns':data[7]}
    index=((msg["side"]+1)*2+msg["type"]).astype("int32")
    ask,bid,trade=jax.lax.switch(index-1,(ask_lim,ask_cancel,ask_cancel,ask_mkt,bid_lim,bid_cancel,bid_cancel,bid_mkt),msg,askside,bidside)
    return (ask,bid),trade

vcond_type_side=jax.vmap(cond_type_side,((0,0),1),0)


def scan_through_entire_array(msg_array,ordersides):
    ordersides,trades=jax.lax.scan(cond_type_side,ordersides,msg_array)
    return ordersides,trades

vscan_through_entire_array=jax.vmap(scan_through_entire_array,(2,(0,0)),0)


@partial(jax.jit,static_argnums=(1,2))
def branch_type_side(data,type,side,askside,bidside):
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
            matchtuple=match_against_ask_orders(askside,msg["quantity"],msg["price"],jnp.ones((5,5))*-1)
            #^(orderside,qtm,price,trade)
            msg["quantity"]=matchtuple[1]
            bids=add_order(bidside,msg)
            return (matchtuple[0],bids),matchtuple[3]
        elif type==2:
            #cancel order on bids side
            return (askside,cancel_order(bidside,msg)),jnp.ones((5,5))*-1
        elif type==3:
            #cancel order on bids side
            return (askside,cancel_order(bidside,msg)),jnp.ones((5,5))*-1
        elif type==4:
            msg["price"]=MAXPRICE
            matchtuple=match_against_ask_orders(askside,msg["quantity"],msg["price"],jnp.ones((5,5))*-1)
            #^(orderside,qtm,price,trade)
            return (matchtuple[0],bidside),matchtuple[3]
    else:
        if type==1:
            #match with bids side
            #add remainder to asks side
            matchtuple=match_against_bid_orders(bidside,msg["quantity"],msg["price"],jnp.ones((5,5))*-1)
            #^(orderside,qtm,price,trade)
            msg["quantity"]=matchtuple[1]
            asks=add_order(askside,msg)
            return (asks,matchtuple[0]),matchtuple[3]
        elif type==2:
            #cancel order on asks side
            return (cancel_order(askside,msg),bidside),jnp.ones((5,5))*-1
        elif type==3:
            #cancel order on asks side
            return (cancel_order(askside,msg),bidside),jnp.ones((5,5))*-1
        elif type==4:
            #set price to 0
            #match with bids side 
            #no need to add remainder
            msg["price"]=0
            matchtuple=match_against_bid_orders(bidside,msg["quantity"],msg["price"],jnp.ones((5,5))*-1)
            #^(orderside,qtm,price,trade)
            return (askside,matchtuple[0]),matchtuple[3]





########Type Functions#############

def bid_lim(msg,askside,bidside):
    #match with asks side
    #add remainder to bids side
    matchtuple=match_against_ask_orders(askside,msg["quantity"],msg["price"],jnp.ones((5,5))*-1)
    #^(orderside,qtm,price,trade)
    msg["quantity"]=matchtuple[1]
    bids=add_order(bidside,msg)
    return matchtuple[0],bids,matchtuple[3]

def bid_cancel(msg,askside,bidside):
    return askside,cancel_order(bidside,msg),jnp.ones((5,5))*-1

def bid_mkt(msg,askside,bidside):
    msg["price"]=MAXPRICE
    matchtuple=match_against_ask_orders(askside,msg["quantity"],msg["price"],jnp.ones((5,5))*-1)
    #^(orderside,qtm,price,trade)
    return matchtuple[0],bidside,matchtuple[3]


def ask_lim(msg,askside,bidside):
    #match with bids side
    #add remainder to asks side
    matchtuple=match_against_bid_orders(bidside,msg["quantity"],msg["price"],jnp.ones((5,5))*-1)
    #^(orderside,qtm,price,trade)
    msg["quantity"]=matchtuple[1]
    asks=add_order(askside,msg)
    return asks,matchtuple[0],matchtuple[3]

def ask_cancel(msg,askside,bidside):
    return cancel_order(askside,msg),bidside,jnp.ones((5,5))*-1

def ask_mkt(msg,askside,bidside):
    #set price to 0
    #match with bids side 
    #no need to add remainder
    msg["price"]=0
    matchtuple=match_against_bid_orders(bidside,msg["quantity"],msg["price"],jnp.ones((5,5))*-1)
    #^(orderside,qtm,price,trade)
    return askside,matchtuple[0],matchtuple[3]



######Helper functions for getting information #######


def get_totquant_at_price(orderside,price):
        return jnp.sum(jnp.where(orderside[:,0]==price,orderside[:,1],0))

get_totquant_at_prices=jax.vmap(get_totquant_at_price,(None,0),0)

@partial(jax.jit,static_argnums=0)
def get_L2_state(N,asks,bids):
    bid_prices=-jnp.unique(-bids[:,0],size=N,fill_value=1)
    ask_prices=jnp.unique(jnp.where(asks[:,0]==-1,MAXPRICE,asks[:,0]),size=N,fill_value=MAXPRICE)
    ask_prices=jnp.where(ask_prices==MAXPRICE,-1,ask_prices)

    bid_quants=get_totquant_at_prices(bids,bid_prices)
    ask_quants=get_totquant_at_prices(asks,ask_prices)
    bid_quants=jnp.where(bid_quants<0,0,bid_quants)
    ask_quants=jnp.where(ask_quants<0,0,ask_quants)
    return jnp.stack((ask_prices,ask_quants,bid_prices,bid_quants),axis=1,dtype=jnp.int32)

@partial(jax.jit,static_argnums=0)
def init_orderside(nOrders=100):
    return (jnp.ones((nOrders,6))*-1).astype("int32")


#TODO: Actually complete this function to not only return dummy vars
def get_initial_orders(bookData,idx_window):
    orderbookLevels=10
    #jax.debug.print("Book Data: \n {}",bookData[idx_window])
    #TODO selecting correct slice based on idx_window
    data=jnp.array(bookData[0]).reshape(int(10*2),2)
    newarr = jnp.zeros((int(orderbookLevels*2),8))
    initOB = newarr \
        .at[:,3].set(data[:,0]) \
        .at[:,2].set(data[:,1]) \
        .at[:,0].set(1) \
        .at[0:orderbookLevels*4:2,1].set(-1) \
        .at[1:orderbookLevels*4:2,1].set(1) \
        .at[:,4].set(INITID) \
        .at[:,5].set(INITID-jnp.arange(0,orderbookLevels*2)) \
        .at[:,6].set(34200) \
        .at[:,7].set(0).astype('int32')
    return initOB


def get_data_messages():
    return jnp.array([[1,-1,200,210000,8888888,8888889,3567,455768],
                        [1,-1,100,210009,8888888,8888890,3577,4567]])



"""def init_msgs_from_l2(book: Union[pd.Series, onp.ndarray]) -> jnp.ndarray:
    """"""
    orderbookLevels = len(book) // 4  # price/quantity for bid/ask
    data = jnp.array(book).reshape(int(orderbookLevels*2),2)
    newarr = jnp.zeros((int(orderbookLevels*2),8))
    initOB = newarr \
        .at[:,3].set(data[:,0]) \
        .at[:,2].set(data[:,1]) \
        .at[:,0].set(1) \
        .at[0:orderbookLevels*4:2,1].set(-1) \
        .at[1:orderbookLevels*4:2,1].set(1) \
        .at[:,4].set(0) \
        .at[:,5].set(job.INITID) \
        .at[:,6].set(34200) \
        .at[:,7].set(0).astype('int32')
    return initOB"""                        