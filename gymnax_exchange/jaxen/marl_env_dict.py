import os, sys, time, dataclasses
from typing import Tuple, Optional, Dict

import jax
import jax.numpy as jnp
import chex
from flax import struct
import jax.tree_util as jtu
from functools import partial

# for debugging
jax.config.update('jax_disable_jit', True)
jax.config.update("jax_log_compiles", False)

sys.path.append(os.path.abspath("/home/duser/AlphaTrade"))

from mm_env import MarketMakingEnv, EnvState as MMState, EnvParams as MMParams
from exec_env import ExecutionEnv, EnvState as EXEState, EnvParams as EXEParams
from gymnax_exchange.jaxen.base_env import BaseLOBEnv, EnvState as BaseState, EnvParams as BaseParams
from gymnax_exchange.jaxob import JaxOrderBookArrays as job

from gymnax_exchange.jaxob.jaxob_config import EnvironmentConfig
from gymnax_exchange.jaxob.jaxob_config import EnvironmentExecutionConfig
from gymnax_exchange.jaxob.jaxob_config import Configuration

# Define a combined (multi–agent) state that extends the base order book state
@struct.dataclass
class MultiAgentState(BaseState):
    # Sub–state for market maker and execution agent.
    mm_state: MMState
    exe_state: EXEState

# Define a combined parameters class.
@struct.dataclass
class MultiAgentParams(BaseParams):
    mm_params: MMParams
    exe_params: EXEParams

# define the MARL environment.
class MARLEnv(BaseLOBEnv):
    def __init__(self,
                 key,
                 alphatradePath: str,
                 window_index: int,
                 episode_time: int,
                 ep_type: str = "fixed_time",
                 mm_trader_id: int = -9999991,
                 exe_trader_id: int = -9999992,
                 exe_reward_lambda: float = 1.0,
                 ):
        # Initialize the base environment
        #jax.debug.print("Initializing MARLEnv: type(alphatradePath) = {}, alphatradePath = {}", type(alphatradePath), alphatradePath)

        # Create config first
        self.cfg = Configuration()
        
        # Pass config to parent class
        super().__init__(self.cfg, key, alphatradePath, window_index, episode_time, ep_type=ep_type)

         # Split the key for the sub-environments:
        key_mm, key_exe = jax.random.split(key, 2)
        
        mm_config = EnvironmentConfig()

        print("Initializing MM environment...")
        # Create the market making sub-env 
        self.mm_env = MarketMakingEnv(
            key=key_mm,
            cfg=mm_config,
            alphatradePath=alphatradePath,
            window_index=window_index,
            episode_time=episode_time,
            trader_unique_id = mm_trader_id,
            ep_type=ep_type
        )
        
        exe_config = EnvironmentExecutionConfig()

        print("Initializing EXE environment...")
        # Create the execution sub-env
        self.exe_env = ExecutionEnv(
            cfg = exe_config,
            key=key_exe,
            alphatradePath=alphatradePath,
            window_index=window_index,
            episode_time=episode_time,
            #max_task_size=exe_task_size,
            rewardLambda=exe_reward_lambda,
            trader_unique_id=exe_trader_id, 
            ep_type=ep_type
        )
        
        self.mm_trader_id = mm_trader_id
        self.exe_trader_id = exe_trader_id
        print("MARL Environment initialized")

    @property
    def default_params(self) -> MultiAgentParams:
        # Get the base parameters from BaseLOBEnv
        base_params = super().default_params
        # Get the sub–env default parameters
        exe_params = self.exe_env.default_params
        mm_params = self.mm_env.default_params
        # Combine them into a MultiAgentParams instance.
        return MultiAgentParams(
            **dataclasses.asdict(base_params),
            mm_params=mm_params,
            exe_params=exe_params
        )

    def reset_env(self, key: chex.PRNGKey, params: MultiAgentParams) -> Tuple[Dict[str, jnp.ndarray], MultiAgentState]:
        # Split keys for each sub–env
        key_mm, key_exe, key = jax.random.split(key, 3)
        mm_obs, mm_state = self.mm_env.reset_env(key_mm, params.mm_params)
        exe_obs, exe_state = self.exe_env.reset_env(key_exe, params.exe_params)
        # The shared base state is taken from mm_state
        base_state = mm_state  
        # Manually copy the base state fields
        multi_state = MultiAgentState(
            ask_raw_orders = base_state.ask_raw_orders,
            bid_raw_orders = base_state.bid_raw_orders,
            trades = base_state.trades,
            init_time = base_state.init_time,
            time = base_state.time,
            customIDcounter = base_state.customIDcounter,
            window_index = base_state.window_index,
            step_counter = base_state.step_counter,
            max_steps_in_episode = base_state.max_steps_in_episode,
            start_index = base_state.start_index,
            # And now add the agent–specific states:
            mm_state = mm_state,
            exe_state = exe_state
        )
        multi_obs = {"market_maker": jnp.array(mm_obs, dtype= jnp.float32), "execution": jnp.array(exe_obs, dtype= jnp.float32)}
        return multi_obs, multi_state


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
            state.init_time[0] + params.episode_time
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
            1
        )
        mm_cnl_msgs_ask = job.getCancelMsgs(
            state.ask_raw_orders,
            self.mm_trader_id,
            self.mm_env.cfg.num_messages_by_agent//4,
            -1
        )
        mm_cnl_msgs = jnp.concatenate([mm_cnl_msgs, mm_cnl_msgs_ask], axis=0)

        jax.debug.print(f"Market Maker action msg: {mm_order_msgs}")
        jax.debug.print(f"Market Maker cancel msg: {mm_cnl_msgs}")

        # Do filtering to net cancellations in MM)
        mm_order_msgs, mm_cnl_msgs = self.mm_env._filter_messages(mm_order_msgs, mm_cnl_msgs)

        # -------------------------------------------------------
        # (C) Build Execution messages
        # -------------------------------------------------------
        exe_raw_action = self.exe_env._reshape_action(actions["execution"],
                                                      state.exe_state,
                                                      params.exe_params,
                                                      key_exe)
        exe_order_msgs = self.exe_env._getActionMsgs(exe_raw_action,
                                                     state.exe_state,
                                                     params.exe_params)
        exe_action_prices = exe_order_msgs[:, 3]  # Get action prices
        
        jax.debug.print(f"Execution messages: {exe_order_msgs}")
        
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
            self.exe_env.n_actions,
            side_for_exe
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
            self.cfg,  
            key,  
            combined_msgs,
            (state.ask_raw_orders, state.bid_raw_orders, trades_reinit),
            self.stepLines
        )
        
        # Forward-fill best prices if necessary:
        new_bestasks = self._ffill_best_prices(new_bestasks[-self.stepLines+1:], state.mm_state.best_asks[-1, 0])
        new_bestbids = self._ffill_best_prices(new_bestbids[-self.stepLines+1:], state.mm_state.best_bids[-1, 0])

       

        # Get features of previous state for mm obvs update
        old_time=state.time
        old_mid_price=state.mm_state.mid_price

         # Update time and ID counter
        final_time = combined_msgs[-1, -2:] + params.time_delay_obs_act
        final_id_ctr = state.customIDcounter + self.mm_env.n_actions + 1  

        jax.debug.print(f"MM num actions: {self.mm_env.n_actions}")

        #---------------------------------------------------------
        #(F) End step functions
        #----------------------------------------------------------
        
        # Market maker end fuction
        (new_asks, new_bids, new_trades), new_id_counter, new_time=self.mm_env.get_episode_end_fn(key_mm,
            new_bestasks, new_bestbids, final_time, new_asks, new_bids, new_trades, state.mm_state, params.mm_params)
        
        # Execution End 
        #Find quant executed
        exe_agent_trades = job.get_agent_trades(new_trades, self.exe_trader_id)
        exe_executions = self.exe_env._get_executed_by_action(exe_agent_trades, actions["execution"], state.exe_state)
        exe_quant_executed_this_step = exe_executions.sum()
        quant_left = state.exe_state.task_to_execute - (state.exe_state.quant_executed + exe_quant_executed_this_step)


        (new_asks, new_bids, new_trades), (new_bestask, new_bestbid), new_id_counter, new_time, mkt_exec_quant, doom_quant = \
            self.exe_env.get_episode_end_fn(key_exe,
                quant_left, new_bestasks[-1], new_bestbids[-1], final_time, new_asks, new_bids, new_trades, state.exe_state, params.exe_params)
        new_bestasks = jnp.concatenate([new_bestasks,new_bestasks[-1:,:] ], axis=0, dtype=jnp.int32)
        new_bestbids = jnp.concatenate([new_bestbids, new_bestbids[-1:,:]], axis=0, dtype=jnp.int32)
        

        # -------------------------------------------------------
        # (G) Compute agent-specific rewards and observations
        # -------------------------------------------------------
        mm_agent_trades = job.get_agent_trades(new_trades, self.mm_trader_id)
        mm_executions = self.mm_env._get_executed_by_action(mm_agent_trades, actions["market_maker"], state,mm_action_prices)
        mm_reward, mm_info = self.mm_env._get_reward(state.mm_state, params.mm_params, mm_agent_trades, new_bestasks, new_bestbids)
        #mm_obs = self.mm_env._get_obs(state.mm_state, params.mm_params)
        mm_obs=self.mm_env.get_observation(state.mm_state, params.mm_params, combined_msgs, mm_action_prices, mm_executions,old_time,old_mid_price)

        exe_agent_trades = job.get_agent_trades(new_trades, self.exe_trader_id)
        exe_reward, exe_info = self.exe_env._get_reward(state.exe_state, params.exe_params, exe_agent_trades)
        exe_obs = self.exe_env._get_obs(state.exe_state, params.exe_params)

        jax.debug.print(f"MM trades: {mm_agent_trades}")
        jax.debug.print(f"EXE trades: {exe_agent_trades}")
        jax.debug.print(f"All Trades: {new_trades}")

        jax.debug.print(f"MM obs: {mm_obs}")
        jax.debug.print(f"EXE obs: {exe_obs}")

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
            inventory=mm_info["end_inventory"],
            total_PnL=state.mm_state.total_PnL + mm_info["PnL"],
            mid_price=mm_info["mid_price"],
            cash_balance=mm_info["cash_balance"],
            price_bid_passive=mm_price_bid_passive,
            quant_bid_passive=mm_quant_bid_passive,
            price_ask_passive=mm_price_ask_passive,
            quant_ask_passive=mm_quant_ask_passive
        )

        # Update EXE state with all fields
        new_exe_state = state.exe_state.replace(
            **new_shared_state,
            prev_action=jnp.vstack([exe_action_prices, actions["execution"]]).T,  # store both prices and quantities
            quant_executed=state.exe_state.quant_executed + exe_info["agentQuant"],
            total_revenue=state.exe_state.total_revenue + exe_info["revenue"],
            drift_return=state.exe_state.drift_return + exe_info["drift"],
            advantage_return=state.exe_state.advantage_return + exe_info["advantage"],
            slippage_rm=exe_info["slippage_rm"],
            price_adv_rm=exe_info["price_adv_rm"],
            price_drift_rm=exe_info["price_drift_rm"],
            vwap_rm=exe_info["vwap_rm"],
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
        done = self.is_terminal(new_state, params)
        jax.debug.print(f"Done: {done}")
        dones = {"market_maker": done, "execution": done, "__all__": done} # ALl of them are the same done
        info = {"market_maker": mm_info, "execution": exe_info}
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


    def action_space(self, params: Optional[MultiAgentParams] = None):
        # Return a dictionary of action spaces
        mm_space = self.mm_env.action_space(params.mm_params if params is not None else None)
        exe_space = self.exe_env.action_space(params.exe_params if params is not None else None)
        return {"market_maker": mm_space, "execution": exe_space}

    def observation_space(self, params: Optional[MultiAgentParams] = None):
        mm_space = self.mm_env.observation_space(params.mm_params if params is not None else None)
        exe_space = self.exe_env.observation_space(params.exe_params if params is not None else None)
        return {"market_maker": mm_space, "execution": exe_space}

    @partial(jax.jit, static_argnums=[0])
    def step(self, key, state, actions, params):
        """Override the parent step method to handle dictionaries."""
        # Call step_env to get the raw results
        obs_st, state_st, rewards, dones, infos = self.step_env(key, state, actions, params)
        
        # If needed, get reset observations (for when episodes terminate)
        key_reset = jax.random.fold_in(key, state.step_counter)
        obs_re, state_re = self.reset_env(key_reset, params)
        
        #  Use tree_map for dictionary handling (they do the same thing in JaxMARL )
        ep_done = dones.get("__all__", self.is_terminal(state_st, params))
        obs = jax.tree_map(
            lambda x, y: jax.lax.select(ep_done, x, y), obs_re, obs_st
        )
        next_state = jax.tree_map(
            lambda x, y: jax.lax.select(ep_done, x, y), state_re, state_st
        )

        #jax.debug.print(f"Obs: {obs}")
        
        return obs, next_state, rewards, dones, infos

# --- Example main function to test the MARL environment ---
if __name__ == "__main__":
    import sys
    import time
    import dataclasses

    try:
        ATFolder = sys.argv[1]
        print("AlphaTrade folder:", ATFolder)
    except:
        ATFolder = "/home/duser/AlphaTrade/training_oneDay/train"
        print("Using default folder:", ATFolder)

    config = {
        "EP_TYPE": "fixed_time",
        "EPISODE_TIME": 300,  # for example, 5 minutes
        "WINDOW_INDEX": 1,
        # sub–env parameters:
        "MM_TRADER_ID": -9999991,
        "MM_REWARD_LAMBDA": 0.0001,
        "MM_ACTION_TYPE": "pure",
        "MM_MAX_TASK_SIZE": 500,
        "EXE_TRADER_ID": -9999992,
        "EXE_REWARD_LAMBDA": 1.0,
        "EXE_TASK_SIZE": 100,
    }

    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)

    # Instantiate the MARL environment.
    env = MARLEnv(
        key = key_reset,
        alphatradePath=ATFolder,
        window_index=config["WINDOW_INDEX"],
        episode_time=config["EPISODE_TIME"],
        ep_type=config["EP_TYPE"],
        mm_trader_id=config["MM_TRADER_ID"],
        exe_trader_id=config["EXE_TRADER_ID"],
        #mm_reward_lambda=config["MM_REWARD_LAMBDA"],
    )
    # Get the default combined parameters.
    print("starting default parameters")
    env_params = env.default_params

    env_params = dataclasses.replace(env_params, episode_time=config["EPISODE_TIME"])

    # Reset the environment.
    obs, state = env.reset_env(key_reset, env_params)
    print("Reset done. Market maker obs:", obs["market_maker"])
    print("Execution obs:", obs["execution"])

    # run a loop that samples random actions for each agent.
    for i in range(1, 20):
        print("=" * 40)
        
        print(f"Step {i}")

        key_step, _ = jax.random.split(key_step, 2)

        #key_policy, _ = jax.random.split(key_policy, 2)
        # Get random actions from each agent's action space.

        key_policy, subkey_mm = jax.random.split(key_policy)
        action_mm = env.mm_env.action_space().sample(subkey_mm)

        key_policy, subkey_exe = jax.random.split(key_policy)
        action_exe = env.exe_env.action_space().sample(subkey_exe)
        #action_mm = env.mm_env.action_space().sample(key_policy)
        #action_exe = env.exe_env.action_space().sample(key_policy)
        actions = {"market_maker": action_mm, "execution": action_exe}
        obs, state, rewards, done, info = env.step(key_step, state, actions, env_params)
        print(f"Actions: {actions}")
        print("Step rewards:", rewards)
        print("Step info:", info)
        print("Market Maker Raw Action:", action_mm.tolist())
        print("Execution Raw Action:", action_exe.tolist())
        print("Done:", done)
        if done["__all__"]:
            print("Episode finished!")
            break
