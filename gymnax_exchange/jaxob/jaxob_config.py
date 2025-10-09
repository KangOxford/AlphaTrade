import gymnax_exchange.jaxob.jaxob_constants as cst
import gymnax_exchange.jaxob.jaxenv_constants as env_cst
import os
from typing import OrderedDict, Tuple,  Literal,Union,List

from dataclasses import dataclass,field





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
    start_resolution: int = 50  # Episodes from data start every n seconds.
    alphatradePath: str = os.path.expanduser("~")
    dataPath: str = os.path.expanduser("~")+"/data"
    stock: str = "AMZN"
    timePeriod: str = "2017Jan_oneday" # Needs to be the appropriate directory name. "2017Jan_oneday", "2024"


@dataclass(frozen=True)
class MarketMaking_EnvironmentConfig():
    # action_space options: "fixed_prices", "fixed_quants", "AvSt", "spread_skew", "directional_trading", "simple"
    action_space: str = "spread_skew"

    # observation_space options: "engineered", "messages", "messages_new_tokenizer", "basic"
    observation_space: str = "engineered"
    #end_fn: Literal["force_market_order", "unwind_ref_price","do_nothing"] = "unwind_ref_price"
    # Values for spread skew action space
    spread_multiplier: float = 3.0 #50.0
    skew_multiplier: float = 5.0 #100.0
    n_ticks_in_book : int = 1
    num_messages_by_agent:int=env_cst.num_messages_by_agent
    num_action_messages_by_agent=2 # will be set automcatically down below
    fixed_quant_value:int=10
    n_actions: int = env_cst.n_actions # Only used for fixed_prices
    debug_mode:bool=False
    time_delay_obs_act:int=0
    normalize:bool=True
    short_name:str="MM" # For agent naming e.g. in the obs dict
    seconds_before_episode_end:int=5
    # Fixed action settings
    fixed_action_setting: bool = False
    fixed_action: int = 0
    #Control for simple market making action space
    sell_buy_all_option: bool= False #Whether selling the entire inventory is possible
    simple_nothing_action: bool = True # Whether or not the simple action space has a nothing action

    # Reward
    inv_penalty: str = "none"  # options: "none", "linear", "quadratic", "threshold"
    reward_space: str = "buy_sell_pnl"  # options: "zero_inv", "pnl", "buy_sell_pnl", "complex", "portfolio_value", "portfolio_value_scaled", "spooner", "spooner_damped", "spooner_scaled", "delta_netWorth","weight_pnl_inventory_pnl"
    reference_price_portfolio_value: str = "mid"  # options: "mid", "best_bid_ask", "near_touch"
    inv_penalty_lambda: float = 1.0
    inv_penalty_quadratic_factor: float = 50.0 #Represents N for penalty = 1/N * (inv ** 2) if quadratic penalty is used
    multiplier_type: str = "tick" # options:  "tick" #DO NOT USE "spread" it is WRONG. 
    clip_reward: bool = False
    based_on_mid_price_of_action: bool = True
    exclude_extreme_spreads: bool= False
    # Weights for complex reward function:
    inventoryPnL_lambda: float = 0.5
    unrealizedPnL_lambda: float = 0.1
    asymmetrically_dampened_lambda: float = 0.8


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
        elif self.action_space == "simple":
            object.__setattr__(self, 'n_actions', 4)
            object.__setattr__(self, 'num_messages_by_agent', 4)
            object.__setattr__(self, 'num_action_messages_by_agent', 2)


@dataclass(frozen=True)
class Execution_EnvironmentConfig():
    n_ticks_in_book : int = 1
    task: str = "random"  # options: "random", "buy", "sell"
    action_type: str = "pure"  # options: "delta", "pure"
    action_space: str = "fixed_quants_complex"  # options: "fixed_quants", "fixed_prices", "fixed_quants_complex", "simplest_case", "fixed_quants_1msg"
    observation_space: str = "engineered"  # options: "engineered", "basic", "simplest_case"
    reward_space: str = "normal"  # options: "normal", "finish_fast", "simplest_case"
    #end_fn:Literal["force_market_order","unwind_FT"]="unwind_FT"
    task_size:int= 500
    n_actions:int=5 # will be set automatically in the post init function
    fixed_quant_value:int=10
    num_messages_by_agent:int=8 # will be set automatically in the post init function
    num_action_messages_by_agent:int=4 # will be set automatically in the post init function
    reward_lambda:float= 0.0
    time_delay_obs_act:int=0
    debug_mode:bool=False
    normalize:bool=True
    short_name:str="EXE"
    seconds_before_episode_end:int=5
    doom_price_penalty: float = 0.1
    larger_far_touch_quant: bool = False
    


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
        elif self.action_space == "simplest_case":
            object.__setattr__(self, 'n_actions', 3)
            object.__setattr__(self, 'num_messages_by_agent', 4) # Includes cancel messages
            object.__setattr__(self, 'num_action_messages_by_agent', 2)
        elif self.action_space == "fixed_quants_1msg":
            object.__setattr__(self, 'n_actions', 5)
            object.__setattr__(self, 'num_messages_by_agent', 2)
            object.__setattr__(self, 'num_action_messages_by_agent', 1)
        elif self.action_space == "twap":
            object.__setattr__(self, 'n_actions', 1)
            object.__setattr__(self, 'num_messages_by_agent', 4)
            object.__setattr__(self, 'num_action_messages_by_agent', 2)




@dataclass(frozen=True)
class World_EnvironmentConfig(JAXLOB_Configuration):
    n_data_msg_per_step: int = 1
    window_selector = -1 # -1 means random window
    ep_type :str = "fixed_steps" # fixed_steps, fixed_time
    episode_time: int = 50 # counted by seconds, 1800s=0.5h or steps
    day_start = 34200  # 09:30
    day_end = 57600  # 16:00
    nOrdersPerSide=100 #100
    nTradesLogged=100
    book_depth=10
    n_ticks_in_book = 10 # Depth of PP actions
    customIDCounter=0
    tick_size=100
    trader_id_range_start=-100 # -1 is reserved for the placeholder in the messages object
    placeholder_order_id = -9
    last_step_seconds = 5
    artificial_trader_id_end_episode = -666666 # Artificial trader id for the trade that is artifically added at the end of the episode (this is not really used)
    artificial_order_id_end_episode = -666666 # Artificial order id for the trade that is artifically added at the end of the episode (this is not really used)
    debug_mode:bool=False
    any_message_obs_space:bool=False # TODO: set this automatically in a post init function based on the obs spaces of each agent type
    order_id_counter_start_when_resetting:int=-200
    shuffle_action_messages:bool=True
    use_pickles_for_init:bool= True
    save_raw_observations:bool=False


@dataclass(frozen=True)
class MultiAgentConfig():
    #world_config: World_EnvironmentConfig = field(default_factory=lambda: World_EnvironmentConfig())
    world_config: World_EnvironmentConfig = World_EnvironmentConfig()

    # list_of_agents_configs: List = field(default_factory=lambda: [
    #     MarketMaking_EnvironmentConfig(),
    #     Execution_EnvironmentConfig()
    # ])

    dict_of_agents_configs: dict = field(default_factory=lambda: dict([
        ("MarketMaking", MarketMaking_EnvironmentConfig()),
        ("Execution", Execution_EnvironmentConfig())
    ]))
    number_of_agents_per_type: list = field(default_factory=lambda: [1,1]) # This is only the default value, we change it in the yaml RL file


    # list_of_agents_configs = [
    # list_of_agents_configs: List =field(default_factory=lambda :[Execution_EnvironmentConfig()])
    #     MarketMaking_EnvironmentConfig(),
    #     #Execution_EnvironmentConfig(),
    #     #MarketMaking_EnvironmentConfig(),
    # ]
    # number_of_agents_per_type = [2]


