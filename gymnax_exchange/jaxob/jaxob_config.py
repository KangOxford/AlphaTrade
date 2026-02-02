import gymnax_exchange.jaxob.jaxob_constants as cst
import gymnax_exchange.jaxob.jaxenv_constants as env_cst
import os
from typing import OrderedDict, Tuple,  Literal,Union,List

from dataclasses import dataclass,field





@dataclass(frozen=True)
class JAXLOB_Configuration:
    maxint : int = cst.MaxInt._64_Bit_Signed.value
    init_id :int = cst.INITID
    book_depth: int = 10
    cancel_mode: int= cst.CancelMode.INCLUDE_INITS.value
    type_4_interpretation: int = cst.Type4Interpretation.IOC.value
    seed: int =cst.SEED
    nTrades : int=cst.NTRADE_CAP
    nOrders : int =cst.NORDER_CAP
    simulator_mode: int = cst.SimulatorMode.GENERAL_EXCHANGE.value
    empty_slot_val: int = cst.EMPTY_SLOT
    debug_mode: bool = False
    check_book_fill: bool = True #No major impact on performance, necessary as full book happens quite often.
    start_resolution: int = 6400  # Episodes from data start every n seconds.
    alphatradePath: str = os.path.expanduser("~")
    dataPath: str = os.path.expanduser("~")+"/data"
    stock: str = "AMZN"
    timePeriod: str = "2024_Dec" # Needs to be the appropriate directory name. 


@dataclass(frozen=True)
class MarketMaking_EnvironmentConfig():
    # Debugging options (incl Simple Act Space)
    debug_mode: bool = False
    short_name: str = "MM"  # For agent naming e.g. in the obs dict
    normalize: bool = True
    clip_reward: bool = False
    exclude_extreme_spreads: bool= False


    fixed_action_setting: bool = False
    fixed_action: int = 0
    simple_nothing_action: bool = True # Whether or not the simple action space has a nothing action
    sell_buy_all_option: bool= False
    based_on_mid_price_of_action: bool = True
    tenth_action: str = "MarketOrder"
    bob_v0: int = 1

    
    # Real Parameters
    action_space: str = "bobRL"    # action_space options: "fixed_prices", "fixed_quants", "AvSt", "spread_skew", "directional_trading", "simple"
    observation_space: str = "engineered"    # observation_space options: "engineered", "messages", "messages_new_tokenizer", "basic"
    reward_function: str = "spooner_asym_damped2"  # options: "zero_inv", "pnl", "buy_sell_pnl", "complex", "portfolio_value", "portfolio_value_scaled", "spooner", "spooner_damped", "spooner_scaled", "delta_netWorth","weight_pnl_inventory_pnl"    
    
    #       Values for action space
    spread_multiplier: float = 3.0 #50.0
    skew_multiplier: float = 5.0 #100.0
    n_ticks_offset: int = 1
    fixed_quant_value: int = 10
    auto_liquidate_threshold: int = 10000 # If abs(inventory) exceeds this value, auto submit an IOC aggro order of alpha*Inventory to reduce inventory
    auto_liquidate_alpha: float = 1.0 # Fraction of inventory to ag

    #       Reward
    unwind_price_penalty: int = 5  # Penalty (in ticks) added to the unwind price at episode end
    inv_penalty: str = "none"  # options: "none", "linear", "quadratic", "threshold"
    volume_traded_bonus: str = "none"  # options: "none", "linear"
    reference_price: str = "mid"  # options: "mid_avg", "mid", "far_touch", "near_touch"
    unwind_price: str = "mid"  # options: "mid_avg","mid", "far_touch"
    inv_penalty_lambda: float = 1.0
    inv_penalty_quadratic_factor: float = 50.0 #Represents N for penalty = 1/N * (inv ** 2) if quadratic penalty is used
    inv_penalty_threshold : float = 10.0 #Threshold for threshold based inventory penalty
    multiplier_type: str = "tick" # options:  "tick" #DO NOT USE "spread" it is WRONG. 
    reward_scaling_quo: float = 1.0
    inventoryPnL_eta: float = 0.6
    inventoryPnL_gamma: float = 0.5

    rebate_bps: float = 10.0  # rebate in bps applied to the trade value for limit order fills (only passive fills)

    #       Weights for complex reward function (skip):
    unrealizedPnL_lambda: float = 0.1
    # asymmetrically_dampened_lambda: float = 0.8
    # AvSt specific reward params
    avst_k_parameter: float = 0.4
    avst_var_parameter: float = 1e-8


    # Not actually implemented yet
    time_delay_obs_act: int = 0

    # Set Automatically in Post Init based on action space.
    n_actions: int = 10
    num_messages_by_agent: int = 4
    num_action_messages_by_agent: int = 2


    def __post_init__(self):
        # Since the class is frozen, we need to use object.__setattr__ to modify n_actions
        # Number of messages includes action messages and cancel messages!
        if self.action_space == "fixed_quants":
            if self.tenth_action == "NA":
                object.__setattr__(self, 'n_actions', 9)
            elif self.tenth_action == "MarketOrder":
                object.__setattr__(self, 'n_actions', 10)
            else:
                raise ValueError(f"Invalid tenth_action {self.tenth_action} for fixed_quants action space")
            object.__setattr__(self, 'num_messages_by_agent', 4)
            object.__setattr__(self, 'num_action_messages_by_agent', 2)
        elif self.action_space == "spread_skew":
            object.__setattr__(self, 'n_actions', 6)
            object.__setattr__(self, 'num_messages_by_agent', 4)
            object.__setattr__(self, 'num_action_messages_by_agent', 2)
        elif self.action_space == "bobStrategy":
            object.__setattr__(self, 'n_actions', 5)
            object.__setattr__(self, 'num_messages_by_agent', 4)
            object.__setattr__(self, 'num_action_messages_by_agent', 2)
        elif self.action_space == "bobRL":
            if self.bob_v0 == 1:
                object.__setattr__(self, 'n_actions', 3)
            elif self.bob_v0 == 2:
                object.__setattr__(self, 'n_actions', 5)
            elif self.bob_v0 == 5:
                object.__setattr__(self, 'n_actions', 11)
            elif self.bob_v0 == 10:
                object.__setattr__(self, 'n_actions', 21)
            else:
                raise ValueError(f"Invalid bob_v0 {self.bob_v0} for bobRL action space")
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
    #Debuggging options
    debug_mode:bool=False
    larger_far_touch_quant: bool = False
    normalize:bool=True
    short_name:str="EXE"
    action_type: str = "pure"  # options: "delta", "pure"


    # Real Parameters
    task: str = "random"  # options: "random", "buy", "sell"
    action_space: str = "fixed_quants_complex"  # options: "fixed_quants", "fixed_prices", "fixed_quants_complex", "simplest_case", "fixed_quants_1msg"
    observation_space: str = "engineered"  # options: "engineered", "basic", "simplest_case"
    reward_function: str = "normal"  # options: "normal", "finish_fast", "simplest_case"
    task_size:int= 600
    n_ticks_in_book : int = 1
    fixed_quant_value:int=10
    reward_lambda:float= 0.0
    reward_scaling_quo: float = 1.0
    doom_price_penalty: int = 5
    reference_price: str = "mid"  # options: "mid", "best_bid_ask", "near_touch"

    #Not functional.. yet
    time_delay_obs_act:int=0
    
    #Set Automatically in Post Init based on action space. 
    n_actions:int=5 # will be set automatically in the post init function
    num_messages_by_agent:int=8 # will be set automatically in the post init function
    num_action_messages_by_agent:int=4 # will be set automatically in the post init function

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
    window_selector: int = -1 # -1 means random window
    ep_type :str = "fixed_steps" # fixed_steps, fixed_time
    episode_time: int = 6400 # counted by seconds, 1800s=0.5h or steps
    day_start: int = 34200  # 09:30
    day_end: int = 57600  # 16:00
    tick_size: int = 100
    trader_id_range_start: int = -100 # -1 is reserved for the placeholder in the messages object
    placeholder_order_id: int = -198
    # last_step_seconds: int = None
    artificial_trader_id_end_episode: int = -199 # Artificial trader id for the trade that is artifically added at the end of the episode (this is not really used)
    artificial_order_id_end_episode: int = -199 # Artificial order id for the trade that is artifically added at the end of the episode (this is not really used)
    any_message_obs_space: bool = False # Returns orderbook L2 state for use in tokenization
    order_id_counter_start_when_resetting: int = -200
    shuffle_action_messages: bool = True
    use_pickles_for_init: bool = True
    save_raw_observations: bool = False




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


    def __post_init__(self):
        # Since the class is frozen, we need to use object.__setattr__ to modify n_actions
        # Number of messages includes action messages and cancel messages!
        for agent_type, config in self.dict_of_agents_configs.items():
            if "message" in config.observation_space:
                object.__setattr__(self.world_config, 'any_message_obs_space', True)


CONFIG_OBJECT_DICT = {"MarketMaking": MarketMaking_EnvironmentConfig,
                      "Execution": Execution_EnvironmentConfig}
if __name__ == "__main__":
    mac = MultiAgentConfig(
        dict_of_agents_configs={"MarketMaking":MarketMaking_EnvironmentConfig(
                                    observation_space="engineered")
                                }
                            )
    print(mac)