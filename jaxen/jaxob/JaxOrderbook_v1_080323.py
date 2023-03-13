import jax.numpy as jnp
import sys
import jax
from jax import lax
from jax import jit
import collections

INITID=-999999

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
    orderlist=orderlist.at[0,:].set(order[2:7])
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
    orderlist=jnp.insert(orderlist,jnp.array(listLocation),jnp.array(order[2:7]),axis=0)
    #Remove last element
    #TODO: if last element is non-null merge with penultimate order (or something else?)
    orderlist=jnp.delete(orderlist,jnp.shape(orderlist)[0]-1,axis=0)
    return orderlist

@jax.jit
def delOrder_2arg(order,orderbook):

    def newID(order,orderbook,loc):
        #The order you're looking for has the INIT ID
        locnew=jnp.where((orderbook[:,:,:,3]==INITID)&(orderbook[:,:,:,1]==order[3]),size=1,fill_value=-1)
        return locnew
    
    orderID=order[5]
    
    idx=jnp.where(orderbook[:,:,:,3]==orderID,size=1,fill_value=-1)
    loc=lax.cond(idx[0][0]==-1,newID,lambda x,y,z:z,order,orderbook,idx)
    orderbook=delOrder_3arg(order,orderbook,loc)
    return orderbook

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
    #Replace -1 entries with inf time (move to end)
    times=jnp.where(times==-1,sys.float_info.max,times)
    #Get sorted indeces and use to sort orderlist.
    out=jnp.argsort(times,axis=0)
    orderlist=orderlist[out,:]
    return orderlist

@jax.jit
def cancelOrder(order,orderbook): 
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
    return orderbook

@jax.jit
def processOrderList(toMatch):
    
    def while_cond(toMatch):
        return (toMatch[0][0,0]!=-1) & (toMatch[1] >0)

    def while_body(toMatch):
        def matchTopOrder(toMatch):
            quantToMatch=toMatch[1]-toMatch[0][0,0]
            orderlist=jnp.delete(toMatch[0],0,axis=0)
            orderlist=jnp.append(orderlist,jnp.ones([1,orderlist.shape[1]])*-1,axis=0)
            return (orderlist,quantToMatch)

        def partialMatchTopOrder(toMatch):
            quantToMatch=jnp.float32(0)
            orderlist=toMatch[0].at[0,0].set(toMatch[0][0,0]-toMatch[1])
            return (orderlist,quantToMatch)
    
        #condition is: quant to match is bigger or equal to quant in first order. DO: remove the top order.
        # else: quant to match is smaller than top order: just reduce the volume. 
        toMatch=lax.cond(toMatch[1]>=toMatch[0][0,0],matchTopOrder,partialMatchTopOrder,toMatch)
        return toMatch
    
    #Aim: given a quantity left to match, run through the orderlist
    #and remove the orders in question until the quant to match is 0 or the whole list is empty.
    toMatch_ret=lax.while_loop(while_cond, while_body, toMatch)
    return toMatch_ret
@jax.jit
def processMKTOrder(order,orderbook):
    def while_cond(toMatch):
        #The best price must have some quant, and the mktorder must still want to match some quant. 
        return (toMatch[0][0,0,0]!=-1) & (toMatch[1] >0)
    
    def while_body(toMatch):
        def list_empty(toMatch):
            orderside=jnp.delete(toMatch[0][0],0,axis=0)
            orderside=jnp.append(orderside,jnp.ones([1,orderside.shape[1],orderside.shape[2]])*-1,axis=0)
            return (orderside,toMatch[1][1])
        
        def list_nonempty(toMatch):
            orderside=toMatch[0][0].at[0,:,:].set(toMatch[1][0])
            return (orderside,toMatch[1][1])
        
        ret=processOrderList((toMatch[0][0,:,:],toMatch[1])) #process top order list.
        
        toMatch=lax.cond(ret[0][0,0]==-1,list_empty,list_nonempty,(toMatch,ret))
        return toMatch
    
    side=order[1]
    quant=order[2]
        
    side=((side+1)/2).astype(int)
    orderside=orderbook[side,:,:,:]
    
    toMatch_ret=lax.while_loop(while_cond,while_body,(orderside,quant))
    orderbook=orderbook.at[side,:,:,:].set(toMatch_ret[0])
    return orderbook
@jax.jit
def processLMTOrder(order,orderbook): #limside should be -1/1
    def while_cond(toMatch):
        #Condition to keep matching: remaining quant at best price, remaining quant to match, price better than lim price.
        default=(toMatch[0][0,0,0]!=-1) & (toMatch[1] >0)
        
        #true: side is 
        cond=lax.cond(toMatch[3]==0,lambda x: x[0][0,0,1]>=x[2],lambda x: x[0][0,0,1]<=x[2],toMatch)
        return default&cond
    
    def while_body(toMatch): #(args: orderside,quant,price,matchside)
        def list_empty(toMatch):
            orderside=jnp.delete(toMatch[0][0],0,axis=0)
            orderside=jnp.append(orderside,jnp.ones([1,orderside.shape[1],orderside.shape[2]])*-1,axis=0)
            return (orderside,toMatch[1][1],toMatch[0][2],toMatch[0][3])#returning orderside, quant,price,side
        
        def list_nonempty(toMatch):
            orderside=toMatch[0][0].at[0,:,:].set(toMatch[1][0])
            return (orderside,toMatch[1][1],toMatch[0][2],toMatch[0][3])
        
        ret=processOrderList((toMatch[0][0,:,:],toMatch[1])) #process top order list. (Args: orderlist, quant)
        
        return_val=lax.cond(ret[0][0,0]==-1,list_empty,list_nonempty,(toMatch,ret))
        return return_val
    
    
    
    matchSide=((-order[1]+1)/2).astype(int)
    limSide=((order[1]+1)/2).astype(int)
    orderside=orderbook[matchSide,:,:,:]
    
    toMatch_ret=lax.while_loop(while_cond,while_body,(orderside,order[2],order[3],limSide))
    orderbook=orderbook.at[matchSide,:,:,:].set(toMatch_ret[0])
    
    order=order.at[2].set(toMatch_ret[1])
    
    #orderbook=lax.cond(toMatch_ret[1]==0,lambda x,y: y,addOrder,*(order,orderbook))
    orderbook=addOrder(order,orderbook)
    
    return orderbook

@jax.jit
def doNothing(order,orderbook):
    return orderbook

@jax.jit
def processOrder(orderbook,order):
    orderbook=lax.switch((order[0]-1).astype(int),[processLMTOrder,cancelOrder,delOrder_2arg,processMKTOrder,doNothing,doNothing,doNothing],order,orderbook)
    return orderbook,0


