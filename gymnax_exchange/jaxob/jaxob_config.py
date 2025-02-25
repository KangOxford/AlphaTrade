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
    reward_space: Literal["zero_inv", "complex", "portfolio_value"] =env_cst.reward_space
    observation_space: Literal["engineered", "messages"] = env_cst.observation_space
    n_ticks_in_book : int = env_cst.n_ticks_in_book





    
    