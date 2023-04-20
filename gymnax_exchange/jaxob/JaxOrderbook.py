import jax.numpy as jnp
import sys
import jax
from jax import lax
import collections

from numpy import float32, int32


INITID=90000000
ORDERSIZE=6

'''Module Name'''


@jax.jit
def convertOrder(dictQuote:collections.OrderedDict):
    order=jnp.array(list(dictQuote.values()))
    return order

@jax.jit
def addOrder(order,orderbook):
    def nonZeroQuant(order,orderbook,idx,bidAsk):
        def addPriceLevel(order,orderbook,idx,bidAsk):
            def continueFunc(order,orderbook,idx,bidAsk,lstidx):
                orderside=orderbook[bidAsk,:,:,:]
                orderside=jnp.insert(orderside,lstidx,orderlist,axis=0)
                orderside=jnp.delete(orderside,jnp.shape(orderside)[0]-1,axis=0)
                orderbook=orderbook.at[bidAsk,:,:,:].set(orderside)
                return orderbook
            
            #Create a new orderlist
            orderlist=newPrice(order,orderbook)
            lstidx=jnp.where((order[1]*orderbook[bidAsk,:,0,1]<order[3]*order[1])|(orderbook[bidAsk,:,0,1]==-1),size=1,fill_value=-1)[0][0]
            orderbook=lax.cond(lstidx==-1,lambda order,orderbook,idx,bidAsk,lstidx: orderbook,continueFunc,*(order,orderbook,idx,bidAsk,lstidx))
            return orderbook
        
        def extendPriceLevel(order,orderbook,idx,bidAsk):
            #Add order to an existing orderlist
            orderlist=add_to_orderlist(order,orderbook,idx,bidAsk)
            orderbook=orderbook.at[bidAsk,idx,:,:].set(orderlist)
            return orderbook
        
        orderbook=lax.cond(idx==-1,addPriceLevel,extendPriceLevel,*(order,orderbook,idx,bidAsk))
        return orderbook
    
    bidAsk=((order[1]+1)/2).astype(int)#buy is 1, sell is 0
    idx=jnp.where(orderbook[bidAsk,:,0,1]==order[3],size=1,fill_value=-1)[0][0]
    orderbook=lax.cond(order[2]==0,lambda order,orderbook,idx,bidAsk: orderbook,nonZeroQuant,*(order,orderbook,idx,bidAsk))
    return orderbook

@jax.jit
def newPrice(order,orderbook):
    orderlist=jnp.ones_like(orderbook[0,0,:,:])*-1
    orderlist=orderlist.at[0,:].set(order[2:8])
    return orderlist

@jax.jit
def add_to_orderlist(order,orderbook,list_index,bidAskidx):
    #Retrieve the appropriate list (Correct side / price)
    orderlist=orderbook[bidAskidx,list_index,:,:]
    #Find location in list where to add the order 
    try:
        listLocation=jnp.where((orderlist[:,4]>order[6]) | (orderlist[:,4]==-1),size=1,fill_value=-1)[0][0]
    except:
        #most likely the list is full and nothing can be added. 
        return orderlist
    #Insert at location
    orderlist=jnp.insert(orderlist,jnp.array(listLocation),jnp.array(order[2:8]),axis=0)
    #Remove last element
    #TODO: if last element is non-null merge with penultimate order (or something else?)
    orderlist=jnp.delete(orderlist,jnp.shape(orderlist)[0]-1,axis=0)
    return orderlist



@jax.jit
def delOrder_3arg(order,orderbook,idx):
    def emptyList(args):
        orderside=delPrice(args[0],*args[1])
        orderbook=args[0].at[args[1][0][0],:,:,:].set(orderside)
        return orderbook
    def nonEmptyList(args):
        orderbook=args[0].at[args[1][0][0],args[1][1][0],:,:].set(args[2])
        return orderbook
    orderID=order[5]
    
    orderlist=del_from_orderlist(orderID,orderbook,*idx)
    
    orderbook=lax.cond(orderlist[0,0]==-1,emptyList,nonEmptyList,(orderbook,idx,orderlist))
    return orderbook

@jax.jit
def delPrice(orderbook,bidAskidx,list_idx,list_loc=0):
    #Retrieve appropriate orderside
    orderside=orderbook[bidAskidx[0],:,:,:]
    
    factor=lax.cond(bidAskidx[0]==0,lambda x: 1,lambda x: -1,bidAskidx[0]) #defines order to sort
    
    #Remove order by replacing with -1 fillers
    orderside=orderside.at[list_idx,:,:].set(-1)
    #Extract prices - which will be used to sort
    prices=orderside[:,0,1]
    #Replace -1 entries with inf time (move to end)
    prices=jnp.where(prices==-1,sys.float_info.max,factor*prices)
    #Get sorted indeces and use to sort orderlist.
    out=jnp.argsort(prices,axis=0)
    orderside=orderside[out,:,:]
    
    return orderside

@jax.jit
def del_from_orderlist(orderID,orderbook,bidAskidx,list_index,listLocation):
    bidAskidx=bidAskidx[0]
    list_index=list_index[0]
    listLocation=listLocation[0]
    #Retrieve the appropriate list (Correct side / price)
    orderlist=orderbook[bidAskidx,list_index,:,:]
    #Remove order by replacing with -1 fillers
    orderlist=orderlist.at[listLocation,:].set(-1)
    #Extract times - which will be used to sort
    times=orderlist[:,4]
    times_ns=orderlist[:,5]
    #Replace -1 entries with inf time (move to end)
    times=jnp.where(times==-1,jnp.iinfo('int32').max,times)
    times_ns=jnp.where(times_ns==-1,jnp.iinfo('int32').max,times_ns)
    #Get sorted indeces and use to sort orderlist.
    out=jnp.lexsort((times_ns,times),axis=0)
    #out=jnp.argsort(times,axis=0)
    orderlist=orderlist[out,:]
    return orderlist



@jax.jit
def processOrderList(toMatch):
    
    def while_cond(toMatch):
        return (toMatch[0][0,0]!=-1) & (toMatch[1] >0)

    def while_body(toMatch):
        def matchTopOrder(toMatch):
            quantToMatch=toMatch[1]-toMatch[0][0,0]
            trade=jnp.array([toMatch[0][0,0],toMatch[0][0,1],toMatch[0][0,3],0,0,0,0])
            #TODO: will still have to fill in agressing ID, trade ID, time
            trades=jnp.delete(toMatch[2],-1,axis=0)
            trades=jnp.insert(trades,0,trade,axis=0)
            orderlist=jnp.delete(toMatch[0],0,axis=0)
            orderlist=jnp.append(orderlist,(jnp.ones([1,orderlist.shape[1]])*-1).astype('int32'),axis=0)
            
            return (orderlist.astype('int32'),quantToMatch.astype('int32'),trades)

        def partialMatchTopOrder(toMatch):
            trade=jnp.array([toMatch[1],toMatch[0][0,1],toMatch[0][0,3],0,0,0,0]) #Quant, Price, Standing ID, Agr ID, Trade ID, Time, Time (ns)
            trades=jnp.delete(toMatch[2],-1,axis=0)
            trades=jnp.insert(trades,0,trade,axis=0)
            quantToMatch=jnp.int32(0)
            orderlist=toMatch[0].at[0,0].set(toMatch[0][0,0]-toMatch[1])
            return (orderlist.astype('int32'),quantToMatch.astype('int32'),trades)
    
        #condition is: quant to match is bigger or equal to quant in first order. DO: remove the top order.
        # else: quant to match is smaller than top order: just reduce the volume. 
        toMatch=lax.cond(toMatch[1]>=toMatch[0][0,0],matchTopOrder,partialMatchTopOrder,toMatch)
        return toMatch
    
    #Aim: given a quantity left to match, run through the orderlist
    #and remove the orders in question until the quant to match is 0 or the whole list is empty.
    toMatch_ret=lax.while_loop(while_cond, while_body, toMatch)
    return toMatch_ret


#LIMIT ORDER - LOBSTER ID = 1
@jax.jit
def processLMTOrder(order,orderbook,trades): #limside should be -1/1
    def while_cond(toMatch):
        #Condition to keep matching: remaining quant at best price, remaining quant to match, price better than lim price.
        default=(toMatch[0][0,0,0]!=-1) & (toMatch[1] >0)
        
        #condition to make sure limit order price is still sufficient to match an order on other book side. 
        cond=lax.cond(toMatch[3]==0,lambda x: x[0][0,0,1]>=x[2],lambda x: x[0][0,0,1]<=x[2],toMatch)
        return default&cond
    
    def while_body(toMatch): #(args: orderside,quant,price,matchside)
        def list_empty(toMatch):
            orderside=jnp.delete(toMatch[0][0],0,axis=0)
            orderside=jnp.append(orderside,(jnp.ones([1,orderside.shape[1],orderside.shape[2]])*-1).astype('int32'),axis=0)
            return (orderside,toMatch[1][1],toMatch[0][2],toMatch[0][3],toMatch[1][2])#returning orderside, quant,price,side
        
        def list_nonempty(toMatch):
            orderside=toMatch[0][0].at[0,:,:].set(toMatch[1][0])
            return (orderside,toMatch[1][1],toMatch[0][2],toMatch[0][3],toMatch[1][2])
        
        ret=processOrderList((toMatch[0][0,:,:],toMatch[1].astype(int),toMatch[4])) #process top order list. (Args: orderlist, quant,trades)
        
        return_val=lax.cond(ret[0][0,0]==-1,list_empty,list_nonempty,(toMatch,ret))
        return return_val
    
    
    
    matchSide=((-order[1]+1)/2).astype(int)
    limSide=((order[1]+1)/2).astype(int)
    orderside=orderbook[matchSide,:,:,:]
    
    toMatch_ret=lax.while_loop(while_cond,while_body,(orderside,order[2],order[3],limSide,trades)) #sidedata to match from,quant,price,side of limOrder
    orderbook=orderbook.at[matchSide,:,:,:].set(toMatch_ret[0])
    trades=toMatch_ret[4]
    trades=trades.at[:,5].set(order[6]).at[:,3].set(order[5]).at[:,6].set(order[7])
    order=order.at[2].set(toMatch_ret[1])
    
    orderbook=addOrder(order,orderbook)
    
    return orderbook,trades

#CANCEL ORDER - LOBSTER ID = 2
@jax.jit
def cancelOrder(order,orderbook,trades): 
    orderID=order[5]
    cancelQuant=order[2]
    #Basically just reducing the quantity posted in a given order.
    orderLoc=jnp.where(orderbook[:,:,:,3]==orderID,size=1,fill_value=-1)

    def newID(order,orderbook,loc):
        #The order you're looking for has the INIT ID
        locnew=jnp.where((orderbook[:,:,:,3]==INITID)&(orderbook[:,:,:,1]==order[3]),size=1,fill_value=-1)
        return locnew
    

    def goodID(order,orderbook,loc):
        orderArr=orderbook[loc][0]
        #TODO: Introduce a check for not finding an orderID. 
        newquant=orderArr[0]-cancelQuant #TODO implement a check to make sure that new quant>0
        
        def contCancel(order,orderbook,loc):
            orderArr=orderbook[loc][0]
            newquant=orderArr[0]-cancelQuant 
            orderArr=orderArr.at[0].set(newquant)
            orderbook=orderbook.at[loc].set(orderArr)
            return orderbook
        
        orderbook=lax.cond(newquant>0,contCancel,delOrder_3arg,order,orderbook,loc)
        return orderbook
    loc=lax.cond(orderLoc[0][0]==-1,newID,lambda x,y,z:z,order,orderbook,orderLoc)
    orderbook=goodID(order,orderbook,loc)
    return orderbook,trades


#DELETE ORDER - LOBSTER ID = 3
@jax.jit
def delOrder_2arg(order,orderbook,trades):

    def newID(order,orderbook,loc,trades):
        #The order you're looking for has the INIT ID
        locnew=jnp.where((orderbook[:,:,:,3]==INITID)&(orderbook[:,:,:,1]==order[3]),size=1,fill_value=-1)
        orderbook,trades=cancelOrder(order,orderbook,trades)
        return orderbook
    
    def goodID(order,orderbook,loc,trades):
        orderbook=delOrder_3arg(order,orderbook,loc)
        return orderbook
    
    orderID=order[5]
    idx=jnp.where(orderbook[:,:,:,3]==orderID,size=1,fill_value=-1)
    orderbook=lax.cond(idx[0][0]==-1,newID,goodID,order,orderbook,idx,trades)
    return orderbook,trades

#MARKET ORDER - LOBSTER ID = 4
@jax.jit
def processMKTOrder(order,orderbook,trades):
    def while_cond(toMatch):
        #The best price must have some quant, and the mktorder must still want to match some quant. 
        return (toMatch[0][0,0,0]!=-1) & (toMatch[1] >0)
    
    def while_body(toMatch):
        def list_empty(toMatch):
            orderside=jnp.delete(toMatch[0][0],0,axis=0)
            orderside=jnp.append(orderside,(jnp.ones([1,orderside.shape[1],orderside.shape[2]])*-1).astype('int32'),axis=0)
            return (orderside,toMatch[1][1],toMatch[1][2])
        
        def list_nonempty(toMatch):
            orderside=toMatch[0][0].at[0,:,:].set(toMatch[1][0])
            return (orderside,toMatch[1][1],toMatch[1][2])
        
        ret=processOrderList((toMatch[0][0,:,:],toMatch[1],toMatch[2])) #process top order list.
        
        toMatch=lax.cond(ret[0][0,0]==-1,list_empty,list_nonempty,(toMatch,ret))
        return toMatch
    
    side=order[1]
    quant=order[2]
        
    side=((side+1)/2).astype(int)
    orderside=orderbook[side,:,:,:]
    
    toMatch_ret=lax.while_loop(while_cond,while_body,(orderside,quant,trades))
    trades=toMatch_ret[2]
    trades=trades.at[:,5].set(order[6]).at[:,3].set(order[5]).at[:,6].set(order[7])
    orderbook=orderbook.at[side,:,:,:].set(toMatch_ret[0])
    return orderbook,trades

#PLACEHOLDER NOTHING - LOBSTER ID = 5,6,7
@jax.jit
def doNothing(order,orderbook,trades):
    return orderbook,trades

@jax.jit
def processLOBSTERexecution(order,orderbook,trades):
    #just flipping the side of the book and submitting as a limorder which should immediately match. 
    order=order.at[1].set((order[1]*-1).astype('int32'))
    return processLMTOrder(order,orderbook,trades)



def processOrder(orderbook,order,tradesLen=5):
    trades=(jnp.ones([tradesLen,7])*-1).astype('int32') #Quant, Price, Standing ID, Agr ID, Trade ID, Time, Time (ns)
    order=order.astype('int32')
    orderbook,trades=lax.switch((order[0]-1).astype(int),[processLMTOrder,cancelOrder,delOrder_2arg,processMKTOrder,doNothing,doNothing,doNothing],order,orderbook,trades)
    return orderbook,trades


i32_orderbook = jax.ShapeDtypeStruct((2,100,100,ORDERSIZE), jnp.dtype('int32'))
i32_order = jax.ShapeDtypeStruct((8,), jnp.dtype('int32'))
i32_scalar=jax.ShapeDtypeStruct((), jnp.dtype('int32'))

processOrder_jitted=jax.jit(processOrder,static_argnames='tradesLen')
processOrder_compiled=processOrder_jitted.lower(i32_orderbook,i32_order ,5).compile()
