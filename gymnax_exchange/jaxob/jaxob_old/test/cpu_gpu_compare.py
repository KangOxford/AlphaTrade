from calendar import c
from pickle import TRUE
import sys
from timeit import timeit
sys.path.append('/Users/sasrey/AlphaTrade')
import gymnax_exchange
import gym_exchange
from time import time

from gym_exchange.data_orderbook_adapter.raw_encoder import RawDecoder, RawEncoder
from gym_exchange.data_orderbook_adapter.decoder import Decoder
from gym_exchange.data_orderbook_adapter.encoder import Encoder
from gym_exchange.data_orderbook_adapter.data_pipeline import DataPipeline
import gymnax_exchange.jaxob.JaxOrderbook as job
from jax import numpy as jnp
from gymnax_exchange.jaxob.jorderbook import OrderBook as JaxOb
from gymnax_exchange.jaxob.jvecorderbook import OrderBook as JaxVecOb

from gym_exchange.orderbook.orderbook import OrderBook as cpuOb
import jax




#Turns flow lists to have side be -1/1 rather than ask/bid
def to_order_flow_lists(flow_lists): 
    '''change side format from bid/ask to 1/-1
    side = -1 if item.side == 'ask' else 1'''
    for flow_list in flow_lists:
        for item in flow_list:
            side = -1 if item.side == 'ask' else 1
            item.side = side
    return flow_lists


decoder = RawDecoder(**DataPipeline()())
encoder = RawEncoder(decoder)
print(decoder.historical_data.shape)
print(gym_exchange.Config.raw_horizon)
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

print(message_array[-10:])
print(message_list[-10:])




ob_jax=JaxOb()
t=time()
trades=ob_jax.process_orders_array(message_array).block_until_ready()
tdelta=time()-t

ob_jax=JaxOb()
t=time()
trades=ob_jax.process_orders_array(message_array).block_until_ready()
tdelta_2ndcall=time()-t

ob_cpu=cpuOb()
t=time()
for msg in message_list:
        ob_cpu.processOrder(msg,True,False)
tdelta_cpu=time()-t

vec_of_ob_jax=[JaxOb() for i in range(100)] 
t=time()
for i in range(100):
    trades=vec_of_ob_jax[i].process_orders_array(message_array).block_until_ready()
tdelta_vec=time()-t

ob_jax_vec=JaxVecOb(100)
batched_messages=jnp.stack([message_array for i in range(100)])
t=time()
ob_jax_vec.process_orders_array(batched_messages)
tdelta_vmap=time()-t


ob_jax_vec=JaxVecOb(100)
batched_messages=jnp.stack([message_array for i in range(100)])
t=time()
ob_jax_vec.process_orders_array(batched_messages)
tdelta_vmap_2nd_call=time()-t


ob_jax_vec=JaxVecOb(2)
batched_messages=jnp.stack([message_array for i in range(2)])
t=time()
ob_jax_vec.process_orders_array(batched_messages)
tdelta_vmap_2=time()-t


ob_jax_vec=JaxVecOb(2)
batched_messages=jnp.stack([message_array for i in range(2)])
t=time()
ob_jax_vec.process_orders_array(batched_messages)
tdelta_vmap_2nd_call_2=time()-t


vec_of_ob_cpu=[cpuOb() for i in range(100)]
t=time()
for i in range(100):
    for msg in message_list:
        vec_of_ob_cpu[i].processOrder(msg,True,False)
tdelta_100cpu=time()-t



print('Time for jax orderbook under lax.scan: ',tdelta)
print('Time for jax orderbook under lax.scan 2nd call: ',tdelta_2ndcall)
print('Time for jax orderbook under 100 times lax.scan : ',tdelta_vec)
print('Time for jax orderbook under vmap call: ',tdelta_vmap)
print('Time for jax orderbook under vmap 2nd call: ',tdelta_vmap_2nd_call)
print('Time for jax orderbook under 2env vmap call: ',tdelta_vmap_2)
print('Time for jax orderbook under 2env vmap 2nd call: ',tdelta_vmap_2nd_call_2)
print('Time for cpu orderbook for loop : ', tdelta_cpu)
print('Time for cpu orderbook for loop 100 times: ', tdelta_100cpu)



#For loops, ignoring for speed.
'''
ob_jax_2=JaxOb()
t=time()
for msg in message_list:
    trades=ob_jax_2.process_order(msg).block_until_ready()
tdelta3=time()-t

ob_jax_3=JaxOb()
t=time()
for msg in message_list:
    trades=ob_jax_3.process_order_comp(msg).block_until_ready()
tdelta4=time()-t 

print('Time for jax orderbook for loop (funct in loop compiled JIT): ', tdelta3)
print('Time for jax orderbook for loop (funct in loop compiled AOT): ', tdelta4)
'''

print(ob_jax_vec.orderbooks_array)

from itertools import zip_longest
cpuOB=jnp.array(list(zip_longest(*ob_cpu.get_L2_state(), fillvalue=-1)))
jaxOB=ob_jax.get_L2_state()
jaxOB=jaxOB.reshape(200,4)

print('CPU orderbook final result:\n', cpuOB)
print('GPU orderbook final result:\n', jaxOB[0:50,:])
size=cpuOB.shape[0]
print('Difference in orderbooks for first 50 levels\n', jaxOB[0:size,:]-cpuOB)
