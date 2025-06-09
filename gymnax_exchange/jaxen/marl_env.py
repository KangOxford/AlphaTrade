import os, sys, time, dataclasses
from typing import Tuple, Optional, Dict
import sys
import time
import dataclasses
import jax
from jax import vmap
import jax.numpy as jnp
import chex
from flax import struct
import jax.tree_util as jtu
from functools import partial
from typing import Any
from gymnax_exchange.utils import utils as util
#from typing import List, Tuple

# for debugging
jax.config.update('jax_disable_jit', False)
jax.config.update("jax_log_compiles", False)

from gymnax_exchange.jaxen.mm_env import MarketMakingAgent
from gymnax_exchange.jaxen.exec_env import ExecutionEnv
from gymnax_exchange.jaxen.base_env import BaseLOBEnv
from gymnax_exchange.jaxen.from_JAXMARL.multi_agent_env import MultiAgentEnv
#from gymnax_exchange.jaxen.from_JAXMARL.spaces import Box, MultiDiscrete, Discrete

from gymnax_exchange.jaxen.StatesandParams import MultiAgentState, MultiAgentParams, LoadedEnvParams, LoadedEnvState, WorldState


from gymnax_exchange.jaxob import JaxOrderBookArrays as job
from gymnax_exchange.jaxob.jaxob_config import MarketMaking_EnvironmentConfig
from gymnax_exchange.jaxob.jaxob_config import Execution_EnvironmentConfig
from gymnax_exchange.jaxob.jaxob_config import MultiAgentConfig





# define the MARL environment.
class MARLEnv(MultiAgentEnv):
    def __init__(self,
                 key,
                 multi_agent_config: MultiAgentConfig,
                 ):
        # Initialize the base environment
        #jax.debug.print("Initializing MARLEnv: type(alphatradePath) = {}, alphatradePath = {}", type(alphatradePath), alphatradePath)

        # Create config first
        self.multi_agent_config = multi_agent_config

        self.num_agents = sum(self.multi_agent_config.number_of_agents_per_type)


        
        super().__init__(num_agents=self.num_agents)


       # Pass config to base class
        self.base_env = BaseLOBEnv(cfg=self.multi_agent_config.world_config, key=key)


        # Split the key for each sub-environments:
        # TODO should we give each sub-env a different key?         for i in range(len(self.world_config.list_of_agents_configs)):
            #key_mm, key_exe = jax.random.split(key, 2)
            #mm_config = MarketMaking_EnvironmentConfig()

        
        self.instance_list=[] # List of different agent types. Each type can have several instances of it
        for agent_type_index in range(len(self.multi_agent_config.list_of_agents_configs)):
            agent_config = self.multi_agent_config.list_of_agents_configs[agent_type_index]
            if isinstance(agent_config, MarketMaking_EnvironmentConfig):
                self.instance_list.append(MarketMakingAgent(cfg=agent_config, world_config=self.multi_agent_config.world_config))
            elif isinstance(agent_config, Execution_EnvironmentConfig):
                self.instance_list.append(ExecutionEnv(cfg=agent_config, world_config=self.multi_agent_config.world_config))
            else:
                raise ValueError(f"Invalid agent type: {i}")

        self.action_spaces = [self.instance_list[i].action_space() for i in range(len(self.instance_list))]
        self.observation_spaces = [self.instance_list[i].observation_space() for i in range(len(self.instance_list))]
                
        num_msg_per_step = self.multi_agent_config.world_config.n_data_msg_per_step
        num_action_msg_per_step_by_all_agents = 0
        for agent_type_index in range(len(self.multi_agent_config.number_of_agents_per_type)):
            agent_config = self.multi_agent_config.list_of_agents_configs[agent_type_index]
            num_agents_per_type = self.multi_agent_config.number_of_agents_per_type[agent_type_index]
            num_msg_per_step += agent_config.num_messages_by_agent * num_agents_per_type
            num_action_msg_per_step_by_all_agents += agent_config.num_action_messages_by_agent * num_agents_per_type

        self.num_msgs_per_step = int(num_msg_per_step)
        self.num_action_msgs_per_step_by_all_agents = int(num_action_msg_per_step_by_all_agents)

        print(f"num_msgs_per_step: {self.num_msgs_per_step}")
        print(f"num_action_msgs_per_step_by_all_agents: {self.num_action_msgs_per_step_by_all_agents}")


        print(self.instance_list)
        print("MARL Environment initialized")

    @property
    def default_params(self) -> MultiAgentParams:
        # Get the base parameters from BaseLOBEnv
        base_params = self.base_env.default_params

        # Get the sub–env default parameters
        params_list = []
        next_trader_id_range_start = self.multi_agent_config.world_config.trader_id_range_start #Start with trader id based on config
        #num_msg_per_step = self.multi_agent_config.world_config.n_data_msg_per_step # start with data msg per step and then add the number of messages per step for each agent

        # Set trader ids and get num_msg_per_step, which both depend on all other agents
        for agent_type_index in range(len(self.multi_agent_config.number_of_agents_per_type)):
            print(f"next_trader_id_range_start: {next_trader_id_range_start}")
            print(f"agent type: {self.multi_agent_config.list_of_agents_configs[agent_type_index]}")
            agent_config = self.multi_agent_config.list_of_agents_configs[agent_type_index]
            num_agents_per_type = self.multi_agent_config.number_of_agents_per_type[agent_type_index]
            agent_params, next_trader_id_range_start = self.instance_list[agent_type_index].default_params(agent_config, next_trader_id_range_start, num_agents_per_type)
            print(f"agent_params: {type(agent_params)}")
            #num_msg_per_step = num_msg_per_step + agent_config.num_messages_by_agent * num_agents_per_type # Sum over all agents of that type
            params_list.append(agent_params)


        # Replace episode_time (#TODO add other world params fields)

        # Combine them into a MultiAgentParams instance.
        return MultiAgentParams(
            loaded_params=base_params, 
            # Add the world fields that are not loaded
            #num_msgs_per_step=num_msg_per_step,
            # add the agent params
            agent_params=params_list
        )

    @partial(jax.jit, static_argnums=(0,))
    def reset_env(self, key: chex.PRNGKey, params: MultiAgentParams) -> Tuple[list[jnp.ndarray], MultiAgentState]:
        #################################
        # Split keys for each agent type
        #################################
        num_agent_types = len(self.instance_list)
        keys = jax.random.split(key, num_agent_types + 1)
        agent_keys = keys[:-1]
        world_key = keys[-1]

        ###########################
        #Reset the World State
        ###########################

        # Get the Load State
        load_state = self.base_env.reset_env(key=world_key, params=params.loaded_params, config=self.multi_agent_config.world_config)

        # Reset all variables in the world state that are not on the Load State
        # For bet bids and ask repeat the inital best bids and ask num of messages times
        best_ask, best_bid = job.get_best_bid_and_ask_inclQuants(self.multi_agent_config.world_config, askside=load_state.ask_raw_orders, bidside=load_state.bid_raw_orders)
        bestbids = jnp.tile(best_bid[None, :], (self.num_msgs_per_step, 1))
        bestasks = jnp.tile(best_ask[None, :], (self.num_msgs_per_step, 1))#
        mid_price = jnp.float32((best_bid[0] + best_ask[0]) / 2)
        print(f"mid_price: {mid_price}")

        # Create the world state
        world_state = WorldState(
            **dataclasses.asdict(load_state),  # copy all fields from the loaded state
            best_bids=bestbids,
            best_asks=bestasks,
            step_counter=0,
            time=load_state.init_time,
            order_id_counter=self.multi_agent_config.world_config.order_id_counter_start_when_resetting,
            mid_price=mid_price,      
            delta_time=0.0,     
        )


        ###########################
        #Reset each agent state
        ###########################

        # multi_obs = {}
        agent_state_list = [] # We are using a list (one for each agent type) of arrays (one element for each agent of that type) instead of a dict (JAXMARL)
        agent_obs_list = []

        
        for config_index, (instance, agent_param, agent_key, agent_config) in enumerate(zip(self.instance_list, params.agent_params, agent_keys, self.multi_agent_config.list_of_agents_configs)):

            vmapped_function = vmap(instance.reset_env, in_axes=(0,None,None,None), out_axes = (0,0))
            agent_obs, agent_state = vmapped_function(agent_param, agent_key, world_state, self.num_msgs_per_step)

            agent_state_list.append(agent_state)
            agent_obs_list.append(agent_obs)

            # Create one key for each agent instance of each agent type (i.e. dict will not be nested like the states list)
            #type_key = f"{agent_config.short_name}_{config_index}"
            #multi_obs[type_key] = agent_obs  # shape: (num_agents_of_this_type, obs_dim)
            
            # to convert to flat dict:
            #for agent_idx, obs in enumerate(agent_obs):
            #    dict_key = f"{agent_config.short_name}_{config_index}_{agent_idx}"
            #    multi_obs[dict_key] = obs
        
        multi_state = MultiAgentState(
            world_state=world_state,
            agent_states=agent_state_list
        )

        return agent_obs_list, multi_state



    @partial(jax.jit, static_argnums=(0,))
    def step_env(self,
                 key: chex.PRNGKey,
                 state: MultiAgentState,
                 actions: list[jnp.ndarray],
                 params: MultiAgentParams,
                 ) -> Tuple[Dict[str, jnp.ndarray], MultiAgentState, Dict[str, float], bool, Dict[str, Dict]]:


        # --------------------------------------------------------------------------------
        # (A) Get the lob state before in case any agent uses the message based obs space
        # --------------------------------------------------------------------------------


        if self.multi_agent_config.world_config.any_message_obs_space == True:
            lob_state_before = job.get_L2_state(
                state.ask_raw_orders,  # Current ask orders
                state.bid_raw_orders,  # Current bid orders
                10,  # Number of levels
                self.cfg  
                )
        else:
            lob_state_before = None




        # -------------------------------------------------------
        # (B) Build External Data Messages (common to all agents)
        # -------------------------------------------------------
        data_messages = self.base_env._get_data_messages(
            params.loaded_params.message_data,
            state.world_state.start_index,
            state.world_state.step_counter,
            state.world_state.init_time[0] + self.multi_agent_config.world_config.episode_time
        )


        # -------------------------------------------------------
        # (C) Get the action and cancel messages for each agent 
        # -------------------------------------------------------

        all_action_msgs_list = [] # One element for each agent type
        all_cancel_msgs_list = [] # One element for each agent type


        for agent_type_index in range(len(self.instance_list)):
            agent_state = state.agent_states[agent_type_index]
            agent_params = params.agent_params[agent_type_index]
            agent_actions = actions[agent_type_index]
            vmapped_function = vmap(self.instance_list[agent_type_index]._get_messages, in_axes=(0,None,0,0), out_axes = (0,0))
            action_msgs, cancel_msgs = vmapped_function(agent_actions, state.world_state, agent_state, agent_params)
            all_action_msgs_list.append(action_msgs)
            all_cancel_msgs_list.append(cancel_msgs)

        #print(f"all action msgs: {all_action_msgs_list}")
       # print(f"all cancel msgs: {all_cancel_msgs_list}")
       # print(f"all action msgs shape: {all_action_msgs_list[0].shape}")
       # print(f"all cancel msgs shape: {all_cancel_msgs_list[0].shape}")

        all_action_msgs = jnp.vstack([x.reshape(-1, x.shape[-1]) for x in all_action_msgs_list])
        all_cancel_msgs = jnp.vstack([x.reshape(-1, x.shape[-1]) for x in all_cancel_msgs_list])













        # Replace order ids in the action messages:

        new_order_ids = jnp.arange( 0 , 0 - self.num_action_msgs_per_step_by_all_agents, -1)
        #print(f"new_order_ids before shift: {new_order_ids}")
        #print("jnp full: ", jnp.full(self.num_action_msgs_per_step_by_all_agents, state.world_state.order_id_counter))

        new_order_ids = new_order_ids + jnp.full(self.num_action_msgs_per_step_by_all_agents, state.world_state.order_id_counter)
        #print(f"new_order_ids after shift: {new_order_ids}")








        all_action_msgs = all_action_msgs.at[:, 4].set(new_order_ids)
        new_order_id_counter = state.world_state.order_id_counter - self.num_action_msgs_per_step_by_all_agents # Used later when we update the state, order ids are counter downwards (negative numbers)

        # Combine action and cancel messages
        combined_msgs = jnp.concatenate([all_cancel_msgs, all_action_msgs, data_messages], axis=0)


        #print(f"actions: {actions}")
        #print(f"best ask prices: {state.world_state.best_asks[-1]}")
        #print(f"best bid prices: {state.world_state.best_bids[-1]}")
        #print(f"combined msgs: {combined_msgs}")
        print(f"all action msgs: {all_action_msgs}")
        





        # -------------------------------------------------------
        # (D) Process combined messages through the order book
        # -------------------------------------------------------

        print("-------------------------------- ")
        print("start processing combined messages")
        print("--------------------------------")

        trades_reinit = (jnp.ones((self.multi_agent_config.world_config.nTradesLogged, 8)) * -1).astype(jnp.int32)
        (new_asks, new_bids, new_trades), (new_bestasks, new_bestbids) = job.scan_through_entire_array_save_bidask(
            self.multi_agent_config.world_config,  
            key,  
            combined_msgs,
            (state.world_state.ask_raw_orders, state.world_state.bid_raw_orders, trades_reinit),
             self.num_msgs_per_step
        )

        print("--------------------------------")
        print("end processing combined messages")
        print("--------------------------------")


        # Forward-fill best prices if necessary:
        new_bestasks = self._ffill_best_prices(new_bestasks, state.world_state.best_asks[-1, 0]) # TODO Do we need this?
        new_bestbids = self._ffill_best_prices(new_bestbids, state.world_state.best_bids[-1, 0])


        #jax.debug.print(f"best bids after ffill: {new_bestbids.shape}")







        final_time = combined_msgs[-1, -2:]
        print(f"final time: {final_time}")


        #---------------------------------------------------------
        #(E) Reward for each agent (part of it is that it changes if the episode is done)
        #----------------------------------------------------------
        
        print(f"new trades: {new_trades}")

        # test for a single agent
        #print("--------------------------------")
        #print("--------------------------------")
        #print("--------------------------------")
        #print("agent params: ", params.agent_params[0])
        #print("agent params: ", util.index_tree(params.agent_params,0))
        #agent_params_test_single = util.index_tree(params.agent_params[1],3)
        #agent_state_test_single = util.index_tree(state.agent_states[1],3)
       # 
       # self.instance_list[1]._get_reward(state.world_state, agent_state_test_single, agent_params_test_single, new_trades, new_bestasks, new_bestbids, final_time)



        print("trades: ", new_trades)

        agent_reward_list = []
        agent_extras_list = []

        for agent_type_index in range(len(self.instance_list)):
            print("agent_type_index: ", agent_type_index)
            agent_state = state.agent_states[agent_type_index]
            agent_params = params.agent_params[agent_type_index]
            vmapped_function = vmap(self.instance_list[agent_type_index]._get_reward, in_axes=(None,0,0,None,None,None,None), out_axes = (0,0))
            reward, extras = vmapped_function(state.world_state, agent_state, agent_params, new_trades, new_bestasks, new_bestbids, final_time)
            agent_reward_list.append(reward)
            agent_extras_list.append(extras)

        print("agent_reward_list: ", agent_reward_list)





        # -------------------------------------------------------
        # (F) Update the world state
        # -------------------------------------------------------

        # Save old values for the message based obs space
        old_time=state.world_state.time
        old_mid_price=state.world_state.mid_price


        # Update other parts of the world state
        new_step_counter = state.world_state.step_counter + 1
        new_mid_price = (new_bestbids[-1, 0] + new_bestasks[-1, 0]) / 2
        new_delta_time = final_time[0] + final_time[1]/1e9 - state.world_state.time[0] - state.world_state.time[1]/1e9


        #print("world state: ", state.world_state)

        # Create new world state
        new_world_state = state.world_state.replace(
            ask_raw_orders=new_asks,
            bid_raw_orders=new_bids,
            trades=new_trades,
            best_asks=new_bestasks,
            best_bids=new_bestbids,
            time=final_time,
            order_id_counter=new_order_id_counter,
            step_counter=new_step_counter,
            mid_price=new_mid_price,
            delta_time=new_delta_time
        )

        #print("new world state: ", new_world_state)
      


        # -------------------------------------------------------
        # (G) Update the agent states
        # -------------------------------------------------------


        print("--------------------------------")
        print("start updating agent states")
        print("--------------------------------")


        new_agent_states_list = []
        new_agent_dones_list = []
        new_agent_infos_list = []

        for agent_type_index in range(len(self.instance_list)):
            print("agent_type_index: ", agent_type_index)
            agent_state = state.agent_states[agent_type_index]
            extras = agent_extras_list[agent_type_index]
            vmapped_function = vmap(self.instance_list[agent_type_index].update_state_and_get_done_and_info, in_axes=(None,0,0), out_axes = (0,0,0))
            states, dones, infos = vmapped_function(new_world_state, agent_state, extras)
            new_agent_states_list.append(states)
            new_agent_dones_list.append(dones)
            new_agent_infos_list.append(infos)
            print(f"agent {agent_type_index} info: {infos}")
            print(f"agent {agent_type_index} done: {dones}")
            print(f"agent {agent_type_index} state: {states}")


        print("new_agent_dones_list: ", new_agent_dones_list)





        # -------------------------------------------------------
        # (H) Get the new overall state
        # -------------------------------------------------------

        new_multi_state = MultiAgentState(
            world_state=new_world_state,
            agent_states=new_agent_states_list
        )

        print("new_multi_state: ", new_multi_state)




        # -------------------------------------------------------
        # (I) Get the done of the world
        # -------------------------------------------------------

        print("dones: ", new_agent_dones_list)

        # Flatten all done flags into a single array
        all_dones_flat = jnp.concatenate(new_agent_dones_list)

        # __all__ is True only if every agent is done
        overall_done = jnp.all(all_dones_flat) # Done if all agents are done

        print("overall_done: ", overall_done)
        print("all_dones_flat: ", all_dones_flat)

        dones = {"__all__": overall_done, "agents": new_agent_dones_list}

        print("dones: ", dones)





        # -------------------------------------------------------
        # (J) Create the info
        # -------------------------------------------------------

        # Create the world info
        print("best asks: ", new_world_state.best_asks)
        print("best bids: ", new_world_state.best_bids)

        # Get the average best ask and bid
        average_best_ask = new_world_state.best_asks[:,0].mean()
        average_best_bid = new_world_state.best_bids[:,0].mean()

        print("average best ask: ", average_best_ask)
        print("average best bid: ", average_best_bid)

        world_info = {
            "window_index":new_world_state.window_index,
            "end_mid_price":new_world_state.mid_price,
            "step_counter":new_world_state.step_counter,
            "time":new_world_state.time,
            "order_id_counter":new_world_state.order_id_counter,
            "best_asks":new_world_state.best_asks,
            "best_bids":new_world_state.best_bids ,
            "average_best_ask":average_best_ask,
            "average_best_bid":average_best_bid,
            "delta_time":new_world_state.delta_time,
            "current_step":new_world_state.step_counter,
        }




        ###debug mode full logging. Ensure this is off by default
        if self.multi_agent_config.world_config.debug_mode==True:
            lob_state = job.get_L2_state(
                                new_world_state.ask_raw_orders,  # Current ask orders
                                new_world_state.bid_raw_orders,  # Current bid orders
                                10,  # Number of levels
                                self.multi_agent_config.world_config  
                                )
            info.update({
                "trades": new_trades,
                "total_msgs": combined_msgs,
                "lob_state": lob_state,
            })


        info = {"world":world_info,"agents":new_agent_infos_list}

        print("info: ", info)




        # -------------------------------------------------------
        # (K) Get the observations for each agent
        # -------------------------------------------------------

        agent_obs_list = []

        for agent_type_index in range(len(self.instance_list)):
            agent_state = new_multi_state.agent_states[agent_type_index]
            agent_params = params.agent_params[agent_type_index]
            agent_config = self.instance_list[agent_type_index].cfg
            vmapped_function = vmap(self.instance_list[agent_type_index].get_observation, in_axes=(None,0,0,None,None,None,None,None), out_axes = (0))
            obs = vmapped_function(new_world_state, agent_state, agent_params, combined_msgs, old_time, old_mid_price, lob_state_before, agent_config.normalize)
            agent_obs_list.append(obs)


        print("agent_obs_list: ", agent_obs_list)








        # TODO Add conditional here that if exec done then set its obs to 0





            
        return agent_obs_list, new_multi_state, agent_reward_list, dones, info













    def _ffill_best_prices(self, prices_quants, last_valid_price):
            def ffill(arr, inval=-1):
                """ Forward fill array values `inval` with previous value """
                def f(prev, x):
                    new = jnp.where(x != inval, x, prev)
                    return (new, new)
                # initialising with inval in case first value is already invalid
                _, out = jax.lax.scan(f, inval, arr)
                return out

            # if first new price is invalid (-1), copy over last price
            prices_quants = prices_quants.at[0, 0:2].set(
                jnp.where(
                    # jnp.repeat(prices_quants[0, 0] == -1, 2),
                    prices_quants[0, 0] == -1,
                    jnp.array([last_valid_price, 0]),
                    prices_quants[0, 0:2]
                )
            )
            # set quantity to 0 if price is invalid (-1)
            prices_quants = prices_quants.at[:, 1].set(
                jnp.where(prices_quants[:, 0] == -1, 0, prices_quants[:, 1])
            )
            # forward fill new prices if some are invalid (-1)
            prices_quants = prices_quants.at[:, 0].set(ffill(prices_quants[:, 0]))
            # jax.debug.print("prices_quants\n {}", prices_quants)
            return prices_quants





    # Overrriding the parent function because we want to vmap over different agents of the same type
    def action_space(self):
        return self.action_spaces
    
    def observation_space(self):
        return self.observation_spaces


    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey, params: MultiAgentParams) -> Tuple[Dict[str, chex.Array], MultiAgentState]:
        """Performs resetting of the environment."""

        if params is None:
            raise ValueError("Params must be provided to reset the environment.")
        else:
            return self.reset_env(key, params)



    # Override the parent step function to handle the list of actions and params object
    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: MultiAgentState,
        actions: list[jnp.ndarray],
        params: MultiAgentParams,
        reset_state: Optional[MultiAgentState] = None,
    ) -> Tuple[Dict[str, chex.Array], MultiAgentState, Dict[str, float], Dict[str, bool], Dict]:
        """Performs step transitions in the environment. Resets the environment if done.
        To control the reset state, pass `reset_state`. Otherwise, the environment will reset randomly."""

        key, key_reset = jax.random.split(key)
        obs_st, states_st, rewards, dones, infos = self.step_env(key = key, state = state, actions = actions, params = params)

        if reset_state is None:
            obs_re, states_re = self.reset(key_reset, params)
        else:
            states_re = reset_state
            obs_re = self.get_obs(states_re) # TODO: this is not implemented yet but i think we dont need it because we do not have the reset state as in input
            raise NotImplementedError("Get obs on the MARL level is not implemented yet")

        # Auto-reset environment based on termination
        states = jax.tree.map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y), states_re, states_st
        )
        obs = jax.tree.map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y), obs_re, obs_st
        )
        return obs, states, rewards, dones, infos


    #@partial(jax.jit, static_argnums=[0])
    #def step(self, key, state, actions, params):
    #    """Override the parent step method to handle dictionaries."""
        # Call step_env to get the raw results
    #    obs_st, state_st, rewards, dones, infos = self.step_env(key, state, actions, params)
        
        # If needed, get reset observations (for when episodes terminate)
    #    key_reset = jax.random.fold_in(key, state.step_counter)
    #    obs_re, state_re = self.reset_env(key_reset, params)
        

    #    #  Use tree_map for dictionary handling (they do the same thing in JaxMARL )
    #    ep_done = dones.get("__all__", self.is_terminal(state_st, params))
    #    obs = jax.tree_map(
    #        lambda x, y: jax.lax.select(ep_done, x, y), obs_re, obs_st
    #    )
    #    next_state = jax.tree_map(
    #        lambda x, y: jax.lax.select(ep_done, x, y), state_re, state_st
    #    )

    #    #jax.debug.print(f"Obs: {obs}")
        
    #    return obs, next_state, rewards, dones, infos



# --- Example main function to test the MARL environment ---
if __name__ == "__main__":

    multi_agent_config = MultiAgentConfig()

    rng = jax.random.PRNGKey(42) # TODO i think this should be changed to the new key function in JAX .key()
    rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)

    # Instantiate the MARL environment.
    env = MARLEnv(
        key=key_reset,
        multi_agent_config=multi_agent_config,
    )
    # Get the default combined parameters.
    print("starting default parameters")
    env_params = env.default_params

    # Reset the environment.
    obs, state = env.reset(key_reset, env_params)
    print("obs", obs)

    # run a loop that samples random actions for each agent.
    for i in range(1, 10):
        print("=" * 40)
        
        print(f"Step {i}")

        key_step, _ = jax.random.split(key_step, 2)

        
        # Get random actions from each agent's action space.
        actions_per_type = []
        key, *subkeys = jax.random.split(key_step, len(multi_agent_config.list_of_agents_configs) + 1)
        subkeys = jnp.array(subkeys)
        for i, (space, num_agents) in enumerate(zip(env.action_spaces, multi_agent_config.number_of_agents_per_type)):
            # Split keys for this agent type
            keys = jax.random.split(subkeys[i], num_agents)
            # Sample actions for all agents of this type
            actions = jax.vmap(space.sample)(keys)
            actions_per_type.append(actions)
        print("actions_per_type:", actions_per_type)


        obs, state, rewards, done, info = env.step(key=key_step, state=state, actions=actions_per_type, params=env_params)

        #DEBUG PRINTS
        #jax.debug.print("EXE info:{}",info["execution"])
        #jax.debug.print("MM info:{}",info["market_maker"])


        
        #print(f"Actions: {actions}")
        #print("Step rewards:", rewards)
        #print("Step info:", info)
        #print("Market Maker Raw Action:", action_mm.tolist())
        #print("Execution Raw Action:", action_exe.tolist())
        #print("Done:", done)
        if done["__all__"]:
            print("Episode finished!")
            break
        

    
    # Set number of environments to batch
     
    #=======================================#
    #=========== VMAP TIMING TEST =========#
    #=======================================#

    enable_vmap = True
    if enable_vmap:
        NUM_ENVS = 1000
        rng = jax.random.PRNGKey(42)

        print("\n" + "="*60)
        print("Starting VMAP timing test loop for MARL")
        print("="*60)

        #---------------------------------------
        # Vectorized Reset
        #---------------------------------------
        print("\n[1] Resetting environments...")
        keys_reset = jax.random.split(rng, NUM_ENVS)
        batched_reset_fn = jax.vmap(env.reset_env, in_axes=(0, None)) # All envs have the same params so second arg is None?

        reset_start = time.time()
        obs, state = batched_reset_fn(keys_reset, env_params)
        reset_end = time.time()
        reset_time = reset_end - reset_start
        print(f"Reset completed in {reset_time:.4f} seconds")

        #---------------------------------------
        # Define VMapped Step Function
        #---------------------------------------
        def step_fn(state, key):
            # For each agent type, sample actions for all agents of that type
            subkeys = jax.random.split(key, len(env.action_spaces))
            actions_per_type = []
            for i, (space, num_agents) in enumerate(zip(env.action_spaces, env.multi_agent_config.number_of_agents_per_type)):
                keys = jax.random.split(subkeys[i], num_agents)
                actions = jax.vmap(space.sample)(keys)
                actions_per_type.append(actions)
            return env.step(key, state, actions_per_type, env_params)

        vmap_step = jax.vmap(step_fn, in_axes=(0, 0))

        #---------------------------------------
        # Rollout Loop
        #---------------------------------------
        print("\n[3] Starting episode rollout...")
        step_counter = jnp.zeros(NUM_ENVS, dtype=int)
        done_flags = jnp.zeros(NUM_ENVS, dtype=bool)
        rng = jax.random.PRNGKey(999)

        rollout_start = time.time()

        def cond_fn(val):
            _, _, done_flags, _ = val
            return jnp.any(~done_flags)

        def body_fn(val):
            state, rng, done_flags, step_counter = val
            rng, *keys = jax.random.split(rng, NUM_ENVS + 1)
            keys = jnp.stack(keys)

            obs, next_state, rewards, done, info = vmap_step(state, keys)

            # Masked state update for active environments
            def masked_update(s, ns):
                mask = done_flags
                while mask.ndim < s.ndim:
                    mask = mask[..., None]
                return jnp.where(mask, s, ns)

            state = jax.tree_map(masked_update, state, next_state)
            done_flags = jnp.logical_or(done_flags, done["__all__"])
            step_counter += jnp.where(done_flags, 0, 1)

            return (state, rng, done_flags, step_counter)

        state, rng, done_flags, step_counter = jax.lax.while_loop(
            cond_fn, body_fn, (state, rng, done_flags, step_counter)
        )

        rollout_end = time.time()
        rollout_time = rollout_end - rollout_start
        total_steps = jnp.sum(step_counter)
        avg_steps_per_env = jnp.mean(step_counter)
        avg_time_per_step = rollout_time / total_steps

        #---------------------------------------
        # Final Stats
        #---------------------------------------
        print("\n[4] Timing Results")
        print("-" * 60)
        print(f"Total Envs:           {NUM_ENVS}")
        print(f"Reset time:           {reset_time:.4f} seconds")
        print(f"Rollout (steps) time: {rollout_time:.4f} seconds")
        print(f"Total steps:          {int(total_steps)}")
        print(f"Avg steps per env:    {avg_steps_per_env:.2f}")
        print(f"Avg time per step:    {avg_time_per_step:.6f} seconds")
        print("="*60)