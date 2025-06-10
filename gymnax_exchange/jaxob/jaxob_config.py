import gymnax_exchange.jaxob.jaxob_constants as cst
import gymnax_exchange.jaxob.jaxenv_constants as env_cst
import jax
import os
from typing import Tuple,  Literal,Union,List

from dataclasses import dataclass

@dataclass(frozen=True)
class JAXLOB_Configuration:
    maxint : int = cst.MaxInt._64_Bit_Signed.value
    init_id :int = cst.INITID
    cancel_mode: int= cst.CancelMode.INCLUDE_INITS.value
    seed: int =cst.SEED
    nTrades : int=cst.NTRADE_CAP
    nOrders : int =cst.NORDER_CAP
    simulator_mode=cst.SimulatorMode.GENERAL_EXCHANGE.value
    empty_slot_val=cst.EMPTY_SLOT
    debug_mode:bool=False
    start_resolution: int = env_cst.start_resolution
    alphatradePath: str = os.path.expanduser("~")
    dataPath: str = os.path.expanduser("~")+"/data" # TODO: rename this to data path or do we need the alphatrade path somewhere else?


@dataclass(frozen=True)
class MarketMaking_EnvironmentConfig():
    action_space: Literal["fixed_prices", "fixed_quants", "AvSt","spread_skew","directional_trading"] =env_cst.action_space
    observation_space: Literal["engineered", "messages", "messages_new_tokenizer", "basic"] = env_cst.observation_space
    end_fn: Literal["force_market_order", "unwind_ref_price","do_nothing"] = env_cst.end_fn
    n_ticks_in_book : int = env_cst.n_ticks_in_book
    num_messages_by_agent:int=env_cst.num_messages_by_agent
    num_action_messages_by_agent=2 # will be set automcatically down below
    fixed_quant_value:int=env_cst.fixed_quant_value
    n_actions: int = env_cst.n_actions # Only used for fixed_prices
    debug_mode:bool=False
    time_delay_obs_act:int=0
    normalize:bool=True
    short_name:str="MM" # For agent naming e.g. in the obs dict
    seconds_before_episode_end:int=5
   
    # Reward
    inv_penalty: Literal["none", "linear", "quadratic"] = env_cst.inv_penalty
    reward_space: Literal["zero_inv", "pnl", "complex", "portfolio_value", "portfolio_value_scaled","spooner","spooner_damped","spooner_scaled","delta_netWorth"] =env_cst.reward_space
    reference_price_portfolio_value: Literal["mid", "best_bid_ask", "near_touch"] =env_cst.reference_price_portfolio_value
    # Weights for complex reward function:
    inventoryPnL_lambda: float =env_cst.inventoryPnL_lambda
    unrealizedPnL_lambda: float =0
    asymmetrically_dampened_lambda: float =env_cst.asymmetrically_dampened_lambda

    def __post_init__(self):
        # Since the class is frozen, we need to use object.__setattr__ to modify n_actions
        # Number of messages includes action messages and cancel messages!
        if self.action_space == "fixed_quants":
            object.__setattr__(self, 'n_actions', 8)
            object.__setattr__(self, 'num_messages_by_agent', 4)
            object.__setattr__(self, 'num_action_messages_by_agent', 2)
        elif self.action_space == "spread_skew":
            object.__setattr__(self, 'n_actions', 6)
            object.__setattr__(self, 'num_messages_by_agent', 4)
            object.__setattr__(self, 'num_action_messages_by_agent', 2)
        elif self.action_space == "directional_trading":
            object.__setattr__(self, 'n_actions', 3)
            object.__setattr__(self, 'num_messages_by_agent', 4)
            object.__setattr__(self, 'num_action_messages_by_agent', 2)
        elif self.action_space == "AvSt":
            object.__setattr__(self, 'n_actions', 8)
            object.__setattr__(self, 'num_messages_by_agent', 4)
            object.__setattr__(self, 'num_action_messages_by_agent', 2)
        elif self.action_space == "fixed_prices":
            object.__setattr__(self, 'num_messages_by_agent', self.n_actions*2)
            object.__setattr__(self, 'num_action_messages_by_agent', self.n_actions)


@dataclass(frozen=True)
class Execution_EnvironmentConfig():
    n_ticks_in_book : int = env_cst.n_ticks_in_book
    task: Literal["random", "buy", "sell"] = "buy"
    action_type: Literal["delta", "pure"] = "pure"
    action_space: Literal["fixed_quants","fixed_prices","fixed_quants_complex"]="fixed_quants"
    observation_space: Literal["engineered"] = "engineered"
    end_fn:Literal["force_market_order","unwind_FT"]="unwind_FT"
    task_size:int=10
    n_actions:int=5
    fixed_quant_value:int=10
    num_messages_by_agent:int=8
    num_action_messages_by_agent:int=4
    reward_lambda:float=1.0
    time_delay_obs_act:int=0
    debug_mode:bool=False
    normalize:bool=True
    short_name:str="EXE"
    seconds_before_episode_end:int=5
    

    def __post_init__(self):
        # Since the class is frozen, we need to use object.__setattr__ to modify n_actions
        # Number of messages includes action messages and cancel messages!
        if self.action_space == "fixed_quants":
            object.__setattr__(self, 'n_actions', 5)
            object.__setattr__(self, 'num_messages_by_agent', 8)
            object.__setattr__(self, 'num_action_messages_by_agent', 4)
        elif self.action_space == "fixed_prices":
            object.__setattr__(self, 'num_messages_by_agent', self.n_actions*2)
            object.__setattr__(self, 'num_action_messages_by_agent', self.n_actions)
        elif self.action_space == "fixed_quants_complex":
            object.__setattr__(self, 'n_actions', 13)
            object.__setattr__(self, 'num_messages_by_agent', 8)
            object.__setattr__(self, 'num_action_messages_by_agent', 4)





@dataclass(frozen=True)
class World_EnvironmentConfig(JAXLOB_Configuration):
    n_data_msg_per_step: int = 100
    window_selector = -1 # -1 means random window
    ep_type = "fixed_time" # fixed_steps, fixed_time
    episode_time = 1800 # counted by seconds, 1800s=0.5h
    day_start = 34200  # 09:30
    day_end = 57600  # 16:00
    nOrdersPerSide=100 #100
    nTradesLogged=100
    book_depth=10
    n_ticks_in_book = 10 # Depth of PP actions
    customIDCounter=0
    tick_size=100
    trader_id_range_start=-2 # -1 is reserved for the placeholder in the messages object
    placeholder_order_id = -99
    last_step_seconds = 5
    artificial_trader_id_end_episode = -666666 # Artificial trader id for the trade that is artifically added at the end of the episode (this is not really used)
    artificial_order_id_end_episode = -666666 # Artificial order id for the trade that is artifically added at the end of the episode (this is not really used)
    debug_mode:bool=False
    any_message_obs_space:bool=False # TODO: set this automatically in a post init function based on the obs spaces of each agent type
    order_id_counter_start_when_resetting:int=-3


@dataclass(frozen=True)
class MultiAgentConfig():
    world_config = World_EnvironmentConfig()

    list_of_agents_configs = [
        MarketMaking_EnvironmentConfig(),
        Execution_EnvironmentConfig(),
        #MarketMaking_EnvironmentConfig(),
    ]
    number_of_agents_per_type = [1,1]


    