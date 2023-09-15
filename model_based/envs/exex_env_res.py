# ============== testing scripts ===============
import jax
import jax.numpy as jnp
import gymnax
import sys
sys.path.append('/Users/sasrey/AlphaTrade')
sys.path.append('/homes/80/kang/AlphaTrade')
# from gymnax_exchange.jaxen.exec_env import ExecutionEnv
from gymnax_exchange.jaxes.jaxob_new import JaxOrderBookArrays as job
import chex
import time

import faulthandler

faulthandler.enable()
print("Num Jax Devices:",jax.device_count(),"Device List:",jax.devices())

chex.assert_gpu_available(backend=None)

#Code snippet to disable all jitting.
from jax import config
# config.update("jax_disable_jit", False)
config.update("jax_disable_jit", True)
# ============== testing scripts ===============



from ast import Dict
from contextlib import nullcontext
from email import message
from random import sample
from re import L
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from gymnax.environments import environment, spaces
from typing import Tuple, Optional
import chex
from flax import struct
from gymnax_exchange.jaxes.jaxob_new import JaxOrderBookArrays as job
from gymnax_exchange.jaxen.exec_env import ExecutionEnv
from gymnax_exchange.jaxen.exec_env import EnvParams, EnvState


    

class RedisualExecutionEnv(ExecutionEnv):
    def __init__(self,alphatradePath,task):
        super().__init__(alphatradePath,task)
        # self.n_actions = 4 # [A, M, P, PP] Agressive, MidPrice, Passive, Second Passive
        self.n_actions = 1 # [A] Agressive
    
    def action_space(
        self, params: Optional[EnvParams] = None
    ) -> spaces.Box:
        """Action space of the environment."""
        return spaces.Box(0,100,(self.n_actions,),dtype=jnp.int32)
    
    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: Dict, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        #Obtain the messages for the step from the message data
        data_messages=job.get_data_messages(params.message_data,state.window_index,state.step_counter)
        #jax.debug.print("Data Messages to process \n: {}",data_messages)

        #Assumes that all actions are limit orders for the moment - get all 8 fields for each action message
        types=jnp.ones((self.n_actions,),jnp.int32)
        sides=-1*jnp.ones((self.n_actions,),jnp.int32) if self.task=='sell' else jnp.ones((self.n_actions),jnp.int32) #if self.task=='buy'
        quants=action #from action space
        
        quant = action + base_action
        # Can only use these if statements because self is a static arg.
        # Done: We said we would do ticks, not levels, so really only the best bid/ask is required -- Write a function to only get those rather than sort the whole array (get_L2) 
        def get_prices(state,task):
            best_ask, best_bid = job.get_best_bid_and_ask(state.ask_raw_orders[-1],state.bid_raw_orders[-1]) # doesnt work
            A = best_bid if task=='sell' else best_ask # aggressive would be at bids
            return [A]

        prices=jnp.asarray(get_prices(state,self.task),jnp.int32)
        # jax.debug.print("Prices: \n {}",prices)
        trader_ids=jnp.ones((self.n_actions,),jnp.int32)*self.trader_unique_id #This agent will always have the same (unique) trader ID
        order_ids=jnp.ones((self.n_actions,),jnp.int32)*(self.trader_unique_id+state.customIDcounter)+jnp.arange(0,self.n_actions) #Each message has a unique ID
        times=state.time+params.time_delay_obs_act
        #Stack (Concatenate) the info into an array 
        
        
        action_msgs=jnp.array([types[0],sides[0],quants[0],prices[0],trader_ids[0],order_ids[0],times[0],times[1]]).reshape(1,8)

        # jax.debug.breakpoint()
        #jax.debug.print("Input to cancel function: {}",state.bid_raw_orders[-1])
        cnl_msgs=job.getCancelMsgs(state.ask_raw_orders[-1] if self.task=='sell' else state.bid_raw_orders[-1],-8999,self.n_fragment_max*self.n_actions,-1 if self.task=='sell' else 1)
        #jax.debug.print("Output from cancel function: {}",cnl_msgs)

        #Add to the top of the data messages
        total_messages=jnp.concatenate([action_msgs,data_messages],axis=0)
        # total_messages=jnp.concatenate([cnl_msgs,action_msgs,data_messages],axis=0)
        # jax.debug.print("Total messages: \n {}",total_messages)

        #Save time of final message to add to state
        time=total_messages[-1:][0][-2:]

        #Process messages of step (action+data) through the orderbook
        #To only ever consider the trades from the last step simply replace state.trades with an array of -1s of the same size. 
        trades_reinit=(jnp.ones((self.nTradesLogged,6))*-1).astype(jnp.int32)

        ordersides=job.scan_through_entire_array_save_states(total_messages,(state.ask_raw_orders[-1,:,:],state.bid_raw_orders[-1,:,:],trades_reinit),self.stepLines) 
        #ordersides=job.scan_through_entire_array_save_states(total_messages,(state.ask_raw_orders,state.bid_raw_orders,state.trades),self.stepLines)
        #Update state (ask,bid,trades,init_time,current_time,OrderID counter,window index for ep, step counter,init_price,trades to exec, trades executed)
        #new_execution=get_exec_quant(ordersides[2],)
        
        # =========ECEC QTY========
        # ------ choice1 ----------
        executed = jnp.where((state.trades[:, 0] > 0)[:, jnp.newaxis], state.trades, 0)
        sumExecutedQty = executed[:,1].sum()
        new_execution = sumExecutedQty
        # CAUTION not same executed with the one in the reward
        # CAUTION the array executed here is calculated from the last state
        # CAUTION while the array executedin reward is calc from the update state in this step
        # ------ choice2 ----------
        # new_execution=10
        # =========================
        # jax.debug.breakpoint()
        
        state = EnvState(*ordersides,state.init_time,time,state.customIDcounter+self.n_actions,state.window_index,state.step_counter+1,state.init_price,state.task_to_execute,state.quant_executed+new_execution)
        # jax.debug.print("Trades: \n {}",state.trades)
        done = self.is_terminal(state,params)
        reward=self.get_reward(state, params)
        #jax.debug.print("Final state after step: \n {}", state)
        
        return self.get_obs(state,params),state,reward,done,{"info":0}
        # return self.get_obs(state, params),state,0,True,{"info":0}




# ============================================================================= #
# ============================================================================= #
# ================================== MAIN ===================================== #
# ============================================================================= #
# ============================================================================= #

if __name__ == "__main__":
    try:
        ATFolder = sys.argv[1]
        print("AlphaTrade folder:",ATFolder)
    except:
        ATFolder = '/homes/80/kang/AlphaTrade'
        
    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)

    env=RedisualExecutionEnv(ATFolder,"sell")
    env_params=env.default_params
    print(env_params.message_data.shape, env_params.book_data.shape)

    start=time.time()
    obs,state=env.reset(key_reset,env_params)
    print("State after reset: \n",state)
    print("Time for reset: \n",time.time()-start)
    print(env_params.message_data.shape, env_params.book_data.shape)
    for i in range(1,100):
        test_action=env.action_space().sample(key_policy)
        print(f"Sampled {i}th actions are: ",test_action)
        start=time.time()
        # jax.debug.breakpoint()
        obs,state,reward,done,info=env.step(key_step, state,test_action, env_params)
        print(f"State after {i} step: \n",state,done,file=open('output.txt','a'))
        print(f"Time for {i} step: \n",time.time()-start)

    # ####### Testing the vmap abilities ########
    
    # enable_vmap=False
    # if enable_vmap:
    #     vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
    #     vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))
    #     vmap_act_sample=jax.vmap(env.action_space().sample, in_axes=(0))

    #     num_envs = 10
    #     vmap_keys = jax.random.split(rng, num_envs)

    #     test_actions=vmap_act_sample(vmap_keys)
    #     print(test_actions)

    #     start=time.time()
    #     obs, state = vmap_reset(vmap_keys, env_params)
    #     print("Time for vmap reset with,",num_envs, " environments : \n",time.time()-start)

    #     start=time.time()
    #     n_obs, n_state, reward, done, _ = vmap_step(vmap_keys, state, test_actions, env_params)
    #     print("Time for vmap step with,",num_envs, " environments : \n",time.time()-start)
