import jax.numpy as jnp
from flax import struct
from typing import Any
import chex



########################################################################################
########################################################################################
# States
########################################################################################
########################################################################################

@struct.dataclass
class LoadedEnvState:
    ask_raw_orders: chex.Array
    bid_raw_orders: chex.Array
    trades: chex.Array
    init_time: chex.Array
    window_index:int
    max_steps_in_episode: int
    start_index: int # This should be here because its the same for all agents, but it changes for all agents when resetting (this is why its not in Params)
    


@struct.dataclass
class WorldState(LoadedEnvState):
    # But everything here that is not loaded from the base config but shared by all agents
    best_bids: jnp.ndarray
    best_asks: jnp.ndarray
    step_counter: int
    time: jnp.ndarray
    customIDcounter: jnp.ndarray
    mid_price:float
    delta_time: float


# Define a combined (multi–agent) state that extends the base order book state
@struct.dataclass
class MultiAgentState():
    # Sub–state for market maker and execution agent.
    world_state: WorldState

    agent_states: list[Any]



@struct.dataclass
class MMEnvState():
    inventory: int
    total_PnL: float
    cash_balance: float


@struct.dataclass
class ExecEnvState():
    prev_action: chex.Array
    prev_executed: chex.Array

    # Execution specific stuff
    init_price: int
    task_to_execute: int
    quant_executed: int
    # Execution specific rewards. 
    total_revenue: float
    drift_return: float
    advantage_return: float
    slippage_rm: float
    price_adv_rm: float
    price_drift_rm: float
    vwap_rm: float
    is_sell_task: int
    trade_duration: float






########################################################################################
########################################################################################
# Params
########################################################################################
########################################################################################

@struct.dataclass
class LoadedEnvParams:
    message_data: chex.Array
    book_data: chex.Array
    init_states_array: chex.Array


# Define a combined parameters class.
# Logic: All the data is in BaseParams. All the things that depend on all agents are added to it (e.g. num_msgs_per_step). The rest stays in the config
@struct.dataclass
class MultiAgentParams():
    loaded_params: LoadedEnvParams

    num_msgs_per_step: int
    agent_params: list[Any]


@struct.dataclass
class MMEnvParams():
    trader_id: chex.Array
    time_delay_obs_act: chex.Array


@struct.dataclass
class ExecEnvParams():
    trader_id: chex.Array
    task_size: chex.Array 
    reward_lambda: chex.Array
    time_delay_obs_act: chex.Array



