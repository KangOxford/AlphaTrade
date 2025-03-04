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
    action_space: Literal["fixed_prices", "fixed_quants", "parameterised"] =env_cst.action_space
    observation_space: Literal["engineered", "messages", "messages_new_tokenizer"] = env_cst.observation_space
    end_fn: Literal["force_market_order", "unwind_mid_price","do_nothing"] = env_cst.end_fn
    n_ticks_in_book : int = env_cst.n_ticks_in_book
    # Reward
    inv_penalty: Literal["none", "linear", "quadratic"] = env_cst.inv_penalty
    reward_space: Literal["zero_inv", "pnl", "complex", "portfolio_value"] =env_cst.reward_space
    reference_price_portfolio_value: Literal["mid", "best_bid_ask"] =env_cst.reference_price_portfolio_value
    n_actions:int=env_cst.n_actions
    
    # Weights for complex reward function:
    inventoryPnL_lambda: float = 0.002
    unrealizedPnL_lambda: float = 0.0
    asymmetrically_dampened_lambda: float = 0.05






    
    