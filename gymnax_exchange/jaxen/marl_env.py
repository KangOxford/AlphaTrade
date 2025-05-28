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
from typing import List, Tuple

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
                
        print("action spacaes:" , self.action_spaces)
        print("observation spaces:" , self.observation_spaces)

        num_msg_per_step = self.multi_agent_config.world_config.n_data_msg_per_step
        for agent_type_index in range(len(self.multi_agent_config.number_of_agents_per_type)):
            agent_config = self.multi_agent_config.list_of_agents_configs[agent_type_index]
            num_agents_per_type = self.multi_agent_config.number_of_agents_per_type[agent_type_index]
            num_msg_per_step += agent_config.num_messages_by_agent * num_agents_per_type

        self.num_msgs_per_step = int(num_msg_per_step)


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

    #@partial(jax.jit, static_argnums=(0,))
    def reset_env(self, key: chex.PRNGKey, params: MultiAgentParams) -> Tuple[List[jnp.ndarray], MultiAgentState]:
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
            customIDcounter=0,
            mid_price=mid_price,      
            delta_time=0.0,     
        )


        ###########################
        #Reset each agent state
        ###########################

        # multi_obs = {}
        agent_state_list = [] # We are using a list (one for each agent type) of arrays (one element for each agent of that type) instead of a dict (JAXMARL)
        agent_obs_list = []



        print("params:", params.agent_params)
        

        
        for config_index, (instance, agent_param, agent_key, agent_config) in enumerate(zip(self.instance_list, params.agent_params, agent_keys, self.multi_agent_config.list_of_agents_configs)):
            print("########################################################")
            print("agent_config:", agent_config)

            vmapped_function = vmap(instance.reset_env, in_axes=(0,None,None,None), out_axes = (0,0))
            agent_obs, agent_state = vmapped_function(agent_param, agent_key, world_state, self.num_msgs_per_step)

            print("agent_obs:", agent_obs.shape)

            agent_state_list.append(agent_state)
            agent_obs_list.append(agent_obs)

            # Create one key for each agent instance of each agent type (i.e. dict will not be nested like the states list)
            #type_key = f"{agent_config.short_name}_{config_index}"
            #multi_obs[type_key] = agent_obs  # shape: (num_agents_of_this_type, obs_dim)
            
            # to convert to flat dict:
            #for agent_idx, obs in enumerate(agent_obs):
            #    dict_key = f"{agent_config.short_name}_{config_index}_{agent_idx}"
            #    multi_obs[dict_key] = obs
        
        print("multi_obs:", agent_obs_list)

        multi_state = MultiAgentState(
            world_state=world_state,
            agent_states=agent_state_list
        )

        return agent_obs_list, multi_state



    def step_env(self,
                 key: chex.PRNGKey,
                 state: MultiAgentState,
                 actions: Dict[str, jnp.ndarray],
                 params: MultiAgentParams
                 ) -> Tuple[Dict[str, jnp.ndarray], MultiAgentState, Dict[str, float], bool, Dict[str, Dict]]:

        # Split keys for each agent (and one extra if needed)
        key_mm, key_exe, key = jax.random.split(key, 3)

        # -------------------------------------------------------
        # (A) Build External Data Messages (common to both agents)
        # -------------------------------------------------------
        data_messages = self._get_data_messages(
            params.message_data,
            state.start_index,
            state.step_counter,
            state.init_time[0] + self.world_config.episode_time
        )

        # -------------------------------------------------------
        # (B) Build Market Maker messages
        # -------------------------------------------------------
        # Use the MM env's message-building functions
        mm_order_msgs = self.mm_env.get_action(actions["market_maker"],
                                                    state.mm_state,
                                                    params.mm_params)
        #mm_order_msgs = self.mm_env._getActionMsgs_fixedQuant(mm_raw_action,
        #                                           state.mm_state,
        #                                           params.mm_params)
        mm_action_prices = mm_order_msgs[:, 3]


        mm_cnl_msgs = job.getCancelMsgs(
            state.bid_raw_orders,  # using the shared order book from the base state
            self.mm_trader_id,
            self.mm_env.cfg.num_messages_by_agent//4,
            1,
            state.time[0],
            state.time[1]
        )
        mm_cnl_msgs_ask = job.getCancelMsgs(
            state.ask_raw_orders,
            self.mm_trader_id,
            self.mm_env.cfg.num_messages_by_agent//4,
            -1,
            state.time[0],
            state.time[1]
        )
        mm_cnl_msgs = jnp.concatenate([mm_cnl_msgs, mm_cnl_msgs_ask], axis=0)

       # jax.debug.print(f"Market Maker action msg: {mm_order_msgs}")
       # jax.debug.print(f"Market Maker cancel msg: {mm_cnl_msgs}")

        # Do filtering to net cancellations in MM)
        mm_order_msgs, mm_cnl_msgs = self.mm_env._filter_messages(mm_order_msgs, mm_cnl_msgs)

        # -------------------------------------------------------
        # (C) Build Execution messages
        # -------------------------------------------------------
        exe_raw_action = self.exe_env._reshape_action(actions["execution"],
                                                      state.exe_state,
                                                      params.exe_params,
                                                      key_exe)
        exe_order_msgs = self.exe_env.get_action(exe_raw_action,
                                                     state.exe_state,
                                                     params.exe_params)
        exe_action_prices = exe_order_msgs[:, 3]  # Get action prices
        exe_action_quants=exe_order_msgs[:,2]
        #jax.debug.print(f"Execution messages: {exe_order_msgs}")
        
        # For execution, decide which side to cancel (depending on task)
        side_for_exe = 1 - state.exe_state.is_sell_task * 2
        raw_order_side = jax.lax.cond(
            state.exe_state.is_sell_task,
            lambda: state.ask_raw_orders,
            lambda: state.bid_raw_orders
        )
        exe_cnl_msgs = job.getCancelMsgs(
            raw_order_side,
            self.exe_trader_id,
            self.exe_env.cfg.num_messages_by_agent//2, #cant be n_actions due to new space
            side_for_exe,
            state.time[0],  # cancel_time
            state.time[1],  # cancel_time_ns
        )
        exe_order_msgs, exe_cnl_msgs = self.exe_env._filter_messages(exe_order_msgs, exe_cnl_msgs)

        # -------------------------------------------------------
        # (D) Combine all agent messages with data messages
        # -------------------------------------------------------
        combined_msgs = jnp.concatenate([
            mm_cnl_msgs,
            mm_order_msgs,
            exe_cnl_msgs,
            exe_order_msgs,
            data_messages
        ], axis=0)

        # -------------------------------------------------------
        # (E) Process combined messages through the order book
        # -------------------------------------------------------

        #jax.debug.print(f"Combined messages: {combined_msgs}")

        trades_reinit = (jnp.ones((self.nTradesLogged, 8)) * -1).astype(jnp.int32)
        (new_asks, new_bids, new_trades), (new_bestbids, new_bestasks) = job.scan_through_entire_array_save_bidask(
            self.world_config,  
            key,  
            combined_msgs,
            (state.ask_raw_orders, state.bid_raw_orders, trades_reinit),
            self.n_data_msg_per_step + self.exe_env.world_config.num_messages_by_agent + self.mm_env.cfg.num_messages_by_agent
        )
        #jax.debug.print(f"New best bids after LOB: {new_bestbids.shape}")
        
        # Forward-fill best prices if necessary:
        new_bestasks = self._ffill_best_prices(new_bestasks[-self.n_data_msg_per_step-self.exe_env.cfg.num_messages_by_agent-self.mm_env.cfg.num_messages_by_agent:], state.mm_state.best_asks[-1, 0]) # TODO this should just be the entire array 
        new_bestbids = self._ffill_best_prices(new_bestbids[-self.n_data_msg_per_step-self.exe_env.cfg.num_messages_by_agent-self.mm_env.cfg.num_messages_by_agent:], state.mm_state.best_bids[-1, 0])

        #jax.debug.print(f"best bids after ffill: {new_bestbids.shape}")

        # Get features of previous state for mm obvs update
        old_time=state.time
        old_mid_price=state.mm_state.mid_price

         # Update time and ID counter
        final_time = combined_msgs[-1, -2:] + params.time_delay_obs_act
        final_id_ctr = state.customIDcounter + self.mm_env.n_actions + 1  

        #jax.debug.print(f"MM num actions: {self.mm_env.n_actions}")

        #---------------------------------------------------------
        #(F) End step functions
        #----------------------------------------------------------
        
        # Market maker end fuction
        (new_asks, new_bids, new_trades), new_id_counter, new_time=self.mm_env.get_episode_end_fn(key_mm,
            new_bestasks, new_bestbids, final_time, new_asks, new_bids, new_trades, state.mm_state, params.mm_params)
        
        # Execution End 
        #Find quant executed
        exe_agent_trades = job.get_agent_trades(new_trades, self.exe_trader_id)
        exe_executions = self.exe_env._get_executed_by_action(exe_agent_trades, actions["execution"], state.exe_state,exe_action_prices)
        exe_executions=jnp.abs(exe_executions)
        exe_quant_executed_this_step = exe_executions[:,1].sum()#new handeling of executions
        quant_left = state.exe_state.task_to_execute - (state.exe_state.quant_executed + exe_quant_executed_this_step)


        (new_asks, new_bids, new_trades), (new_bestask, new_bestbid), new_id_counter, new_time, mkt_exec_quant, doom_quant = \
            self.exe_env.get_episode_end_fn(key_exe,
                quant_left, new_bestasks[-1], new_bestbids[-1], final_time, new_asks, new_bids, new_trades, state.exe_state, params.exe_params)
        #new_bestasks = jnp.concatenate([new_bestasks,new_bestasks[-1:,:] ], axis=0, dtype=jnp.int32)
        #new_bestbids = jnp.concatenate([new_bestbids, new_bestbids[-1:,:]], axis=0, dtype=jnp.int32)
        
        #jax.debug.print(f"best bids after final ep: {new_bestbids.shape}")


        # -------------------------------------------------------
        # (G) Compute agent-specific rewards and observations
        # -------------------------------------------------------
        mm_agent_trades = job.get_agent_trades(new_trades, self.mm_trader_id)
        mm_executions = self.mm_env._get_executed_by_action(mm_agent_trades, actions["market_maker"], state,mm_action_prices)
        mm_executions=jnp.abs(mm_executions) #check incase neg quant
        mm_reward, mm_extras = self.mm_env._get_reward(state.mm_state, params.mm_params, mm_agent_trades, new_bestasks, new_bestbids)
        #mm_obs = self.mm_env._get_obs(state.mm_state, params.mm_params)
        mm_obs=self.mm_env.get_observation(state.mm_state, params.mm_params, combined_msgs, mm_action_prices, mm_executions,old_time,old_mid_price)

        exe_agent_trades = job.get_agent_trades(new_trades, self.exe_trader_id)
        exe_reward, exe_extras = self.exe_env._get_reward(state.exe_state, params.exe_params, exe_agent_trades)
        exe_obs = self.exe_env._get_obs(state.exe_state, params.exe_params)

        #jax.debug.print(f"MM trades: {mm_agent_trades}")
        #jax.debug.print(f"EXE trades: {exe_agent_trades}")
        #jax.debug.print(f"All Trades: {new_trades}")

        #jax.debug.print(f"MM obs: {mm_obs}")
        #jax.debug.print(f"EXE obs: {exe_obs}")

        # -------------------------------------------------------
        # (H) Update the multi–agent state
        # -------------------------------------------------------
        # Update the shared base state fields
        base_state = state  
        delta_time = final_time[0] + final_time[1]/1e9 - state.time[0] - state.time[1]/1e9
        new_shared_state = {
            "ask_raw_orders": new_asks,
            "bid_raw_orders": new_bids,
            "trades": new_trades,
            "time": final_time,
            "customIDcounter": final_id_ctr,
            "best_asks": new_bestasks,
            "best_bids": new_bestbids,
            "step_counter": state.step_counter + 1,
            "delta_time": delta_time
        }

        # Calculate MM-specific state updates
        mm_price_bid_passive, mm_quant_bid_passive, mm_price_ask_passive, mm_quant_ask_passive = self.mm_env._get_pass_price_quant(state.mm_state)

        # Calculate EXE-specific state updates
        exe_price_passive_2, exe_quant_passive_2 = self.exe_env._get_pass_price_quant(state.exe_state)
        exe_trade_duration_step = (jnp.abs(exe_agent_trades[:, 1]) / state.exe_state.task_to_execute * (exe_agent_trades[:, -2] - state.init_time[0])).sum()
        exe_trade_duration = state.exe_state.trade_duration + exe_trade_duration_step

        # Update MM state with all fields
        new_mm_state = state.mm_state.replace(
            **new_shared_state,
            inventory=mm_extras["end_inventory"],
            total_PnL=state.mm_state.total_PnL + mm_extras["PnL"],
            mid_price=mm_extras["mid_price"],
            cash_balance=mm_extras["cash_balance"],
            price_bid_passive=mm_price_bid_passive,
            quant_bid_passive=mm_quant_bid_passive,
            price_ask_passive=mm_price_ask_passive,
            quant_ask_passive=mm_quant_ask_passive
        )

        # Update EXE state with all fields
        new_exe_state = state.exe_state.replace(
            **new_shared_state,
            prev_action=jnp.vstack([exe_action_prices, exe_action_quants]).T,  # store both prices and quantities+> action no longer = quant
            quant_executed=state.exe_state.quant_executed + exe_extras["agentQuant"],
            total_revenue=state.exe_state.total_revenue + exe_extras["revenue"],
            drift_return=state.exe_state.drift_return + exe_extras["drift"],
            advantage_return=state.exe_state.advantage_return + exe_extras["advantage"],
            slippage_rm=exe_extras["slippage_rm"],
            price_adv_rm=exe_extras["price_adv_rm"],
            price_drift_rm=exe_extras["price_drift_rm"],
            vwap_rm=exe_extras["vwap_rm"],
            trade_duration=exe_trade_duration,
            price_passive_2=exe_price_passive_2,
            quant_passive_2=exe_quant_passive_2
        )

        new_state = MultiAgentState(
            ask_raw_orders=new_asks,
            bid_raw_orders=new_bids,
            trades=new_trades,
            init_time=state.init_time,
            time=final_time,
            customIDcounter=final_id_ctr,
            window_index=state.window_index,
            step_counter=state.step_counter + 1,
            max_steps_in_episode=state.max_steps_in_episode,
            start_index=state.start_index,
            mm_state=new_mm_state,
            exe_state=new_exe_state
        )

        obs = {"market_maker": mm_obs, "execution": exe_obs}
        rewards = {"market_maker": mm_reward, "execution": exe_reward}
        mm_done= self.mm_env.is_terminal(state.mm_state,params.mm_params)
        exec_done=self.exe_env.is_terminal(state.exe_state,params.exe_params)
        done = jnp.logical_and(mm_done, exec_done)
        #jax.debug.print("Done: {}",done)
        dones = {"market_maker": mm_done, "execution": exec_done, "__all__": done} # ALl of them are the same done

        #Get infos:
        exe_info = {
            "window_index": new_state.exe_state.window_index,
            "total_revenue": new_state.exe_state.total_revenue,
            "quant_executed": new_state.exe_state.quant_executed,
            "task_to_execute": new_state.exe_state.task_to_execute,
            "average_price": jnp.nan_to_num(new_state.exe_state.total_revenue 
                                            / new_state.exe_state.quant_executed, 0.0),
            "mid_price":((new_state.exe_state.best_bids[:, 0] + new_state.exe_state.best_asks[:, 0]) // 2).mean(),
            "current_step": new_state.exe_state.step_counter,
            "done": done,
            "slippage_rm": new_state.exe_state.slippage_rm,
            "price_adv_rm": new_state.exe_state.price_adv_rm,
            "price_drift_rm": new_state.exe_state.price_drift_rm,
            "vwap_rm": new_state.exe_state.vwap_rm,
            "advantage_reward": new_state.exe_state.advantage_return,
            "drift_reward": new_state.exe_state.drift_return,
            "drift":exe_extras["drift"],
            "trade_duration": new_state.exe_state.trade_duration,
            "mkt_forced_quant": mkt_exec_quant + doom_quant,
            "doom_quant": doom_quant,
            "is_sell_task": new_state.exe_state.is_sell_task,
        }
        average_best_ask = state.mm_state.best_asks[-100:].mean(axis=0)[0]# // self.tick_size) * self.tick_size)
        average_best_bid = state.mm_state.best_bids[-100:].mean(axis=0)[0]#// self.tick_size) * self.tick_size)
        mm_info = {
            "reward":mm_reward,
            "reward_portfolio_value":mm_extras["reward_portfolio_value"],
            "reward_complex":mm_extras["reward_complex"],
            "reward_spooner":mm_extras[ "reward_spooner"],
            "reward_spooner_damped":mm_extras["reward_spooner_damped"],
            "reward_spooner_scaled":mm_extras[ "reward_spooner_scaled"],
            "reward_delta_netWorth":mm_extras["reward_delta_netWorth"],
            "window_index": new_state.mm_state.window_index,
            "total_PnL": new_state.mm_state.total_PnL,                           
            "current_step": new_state.mm_state.step_counter,
            "done": done,
            "time_seconds":new_state.mm_state.time[0],
            "inventory": new_state.mm_state.inventory,
            "market_share":mm_extras["market_share"],
            "buyPnL":mm_extras["buyPnL"],
            "scaledInventoryPnL":mm_extras["scaledInventoryPnL"],
            "netWorth":mm_extras["netWorth"],
            "sellPnL":mm_extras["sellPnL"],
            "buyQuant":mm_extras["buyQuant"],
            "sellQuant":mm_extras["sellQuant"],
            "inventoryValue":mm_extras["inventoryValue"],
            "other_exec_quants":mm_extras["other_exec_quants"],
            "averageMidprice":mm_extras["averageMidprice"],
            "average_best_bid":average_best_bid,
            "average_best_ask":average_best_ask,
            "end_mid_price":mm_extras["mid_price"],
            "Step_PnL":mm_extras["PnL"],
            "action_prices":mm_action_prices,
            "InventoryPnL":mm_extras["InventoryPnL"],
            "approx_realized_pnl":mm_extras["approx_realized_pnl"],
            "approx_unrealized_pnl": mm_extras["approx_unrealized_pnl"]
        } 
        if self.world_config.debug_mode==False:
            info = {"market_maker": mm_info, "execution": exe_info}

        ###debug mode full logging. Ensure this is off by default
        if self.world_config.debug_mode==True:
            lob_state = job.get_L2_state(
                                new_state.ask_raw_orders,  # Current ask orders
                                new_state.bid_raw_orders,  # Current bid orders
                                10,  # Number of levels
                                self.world_config  
                                )
            info = {"market_maker": mm_info, "execution": exe_info,
                "trades":new_trades,
                "total_msgs":combined_msgs,
                "lob_state":lob_state,}
            
        return obs, new_state, rewards, dones, info

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
    obs, state = env.reset_env(key_reset, env_params)
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


        obs, state, rewards, done, info = env.step(key_step, state, actions, env_params)

        #DEBUG PRINTS
        #jax.debug.print("EXE info:{}",info["execution"])
        #jax.debug.print("MM info:{}",info["market_maker"])
        jax.debug.print("market maker reward:{}",rewards["market_maker"])
        jax.debug.print("MM info:{}",info["market_maker"]["reward"])

        
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

    enable_vmap = False
    if enable_vmap:
        NUM_ENVS = 1000
        rng = jax.random.PRNGKey(42)

        print("\n" + "="*60)
        print("Starting VMAP timing test loop for MRL")
        print("="*60)

        #---------------------------------------
        # Vectorized Reset
        #---------------------------------------
        print("\n[1] Resetting environments...")
        keys_reset = jax.random.split(rng, NUM_ENVS)
        batched_reset_fn = jax.vmap(env.reset_env, in_axes=(0, None))

        reset_start = time.time()
        obs, state = batched_reset_fn(keys_reset, env_params)
        reset_end = time.time()
        reset_time = reset_end - reset_start
        print(f"Reset completed in {reset_time:.4f} seconds")

        #---------------------------------------
        # Prepare Dummy Actions
        #---------------------------------------
        print("\n[2] Preparing dummy actions...")
        dummy_action_mm = env.mm_env.action_space().sample(jax.random.PRNGKey(0))
        dummy_action_exe = env.exe_env.action_space().sample(jax.random.PRNGKey(1))

        action_mm = jnp.zeros_like(dummy_action_mm)
        action_exe = jnp.zeros_like(dummy_action_exe)

        #---------------------------------------
        # Define VMapped Step Function
        #---------------------------------------
        def step_fn(state, key):
            actions = {
                "market_maker": action_mm,
                "execution": action_exe
            }
            return env.step(key, state, actions, env_params)

        vmap_step = jax.vmap(step_fn, in_axes=(0, 0))

        #---------------------------------------
        # Rollout Loop
        #---------------------------------------
        print("\n[3] Starting episode rollout...")
        max_steps = config["EPISODE_TIME"]
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