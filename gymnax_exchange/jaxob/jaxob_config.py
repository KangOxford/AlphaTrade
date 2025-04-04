import gymnax_exchange.jaxob.jaxob_constants as cst
import gymnax_exchange.jaxob.jaxenv_constants as env_cst
import jax
from typing import Tuple,  Literal,Union,List

from dataclasses import dataclass

@dataclass(frozen=True)
class Configuration:
    maxint : int = cst.MaxInt._64_Bit_Signed.value
    init_id :int = cst.INITID
    cancel_mode: int= cst.CancelMode.CANCEL_UNIFORM_AND_LARGE.value
    seed: int =cst.SEED
    nTrades : int=cst.NTRADE_CAP
    nOrders : int =cst.NORDER_CAP
    simulator_mode=cst.SimulatorMode.GENERAL_EXCHANGE.value
    empty_slot_val=cst.EMPTY_SLOT

@dataclass(frozen=True)
class EnvironmentConfig(Configuration):
    action_space: Literal["fixed_prices", "fixed_quants", "AvSt","spread_skew","directional_trading"] =env_cst.action_space
    observation_space: Literal["engineered", "messages", "messages_new_tokenizer"] = env_cst.observation_space
    end_fn: Literal["force_market_order", "unwind_ref_price","do_nothing"] = env_cst.end_fn
    n_ticks_in_book : int = env_cst.n_ticks_in_book
    num_messages_by_agent:int=env_cst.num_messages_by_agent
    num_action_messages_by_agent=2
    
    # Reward
    inv_penalty: Literal["none", "linear", "quadratic"] = env_cst.inv_penalty
    reward_space: Literal["zero_inv", "pnl", "complex", "portfolio_value","spooner","spooner_damped","spooner_scaled","delta_netWorth"] =env_cst.reward_space
    reference_price_portfolio_value: Literal["mid", "best_bid_ask"] =env_cst.reference_price_portfolio_value
    
    fixed_quant_value:int=env_cst.fixed_quant_value
    n_actions: int = env_cst.n_actions
    
    # Weights for complex reward function:
    inventoryPnL_lambda: float = 0.6
    unrealizedPnL_lambda: float = 0.0
    asymmetrically_dampened_lambda: float = 0.05

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
            object.__setattr__(self, 'n_actions', 2)
            object.__setattr__(self, 'num_messages_by_agent', 4)
            object.__setattr__(self, 'num_action_messages_by_agent', 2)
        elif self.action_space == "fixed_prices":
            object.__setattr__(self, 'num_messages_by_agent', self.n_actions*2)
            object.__setattr__(self, 'num_action_messages_by_agent', self.n_actions)


@dataclass(frozen=True)
class EnvironmentExecutionConfig(Configuration):
    n_ticks_in_book : int = env_cst.n_ticks_in_book
    task: Literal["random", "buy", "sell"]="buy"
    action_type: Literal["delta", "pure"]="pure"
    action_space: Literal["fixed_quants","fixed_prices"]="fixed_quants"
    end_fn:Literal["force_market_order","unwind_FT"]="unwind_FT"
    max_task_size:int=500
    n_actions:int=4
    fixed_quant_value=10
    num_messages_by_agent=8####make this and the mm one programtic..
    




    
    