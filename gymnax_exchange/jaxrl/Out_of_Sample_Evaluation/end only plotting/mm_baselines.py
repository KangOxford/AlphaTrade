# docker run -it --rm --gpus '"device=7"' -v $(pwd):/app -v $(pwd)/../cache:/app/cache --name ${USER}_lcc ${USER}_lc python -m scripts.rl_test

# tokens are 0: pad, 1, 2: actions, 3 -> 258: observations
import jax
import sys
import os
sys.path.append(os.path.abspath('/home/duser/AlphaTrade')) 
import jax.numpy as jnp
import flax
import datetime
import numpy as np
import optax
import time
from dataclasses import dataclass

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

from typing import Sequence, NamedTuple, Any, Dict, Callable, Optional

import distrax
import gymnax
import functools
from gymnax.environments import spaces
from gymnax_exchange.jaxrl.utils import FlattenObservationWrapper, LogWrapper
from jax._src import dtypes
from gymnax_exchange.jaxen.mm_env import MarketMakingEnv 
from gymnax_exchange.jaxob.jaxob_config import EnvironmentConfig
#import flax
from jax.lib import xla_bridge 
print(xla_bridge.get_backend().platform)
#Code snippet to disable all jitting.
from jax import config
config.update("jax_disable_jit", False) 
# config.update("jax_disable_jit", True)
config.update("jax_check_tracer_leaks", False) #finds a whole assortment of leaks if true... bizarre.
import datetime
import gymnax_exchange.utils.colorednoise as cnoise
jax.numpy.set_printoptions(linewidth=250)
import dataclasses
import jax
import jax.numpy as jnp
#import optax
import distrax
from flax import serialization
import dataclasses
from flax.core import frozen_dict


from jax_rwkv.src.auto import get_rand_model
from gymnax_exchange.jaxrl.rl_processing import get_ppo_agent, calculate_gae, get_jit_ppo, PAD_FLAG, OBS_FLAG, ACT_FLAG
#from utils.jstring import JString



import wandb


wandbOn = True # False
if wandbOn:
    import wandb

"""MMMW baseline agent well compare against"""
class LowRegretMetaAgent:
    def __init__(self, num_strategies=8, eta=0.01, initial_volume=100):
        self.N = num_strategies
        self.eta = eta
        self.weights = jnp.ones(self.N) / self.N  # w₁(b) = 1/N
        self.volumes = initial_volume
        self.prev_values = jnp.zeros(self.N)
        self.current_values = jnp.zeros(self.N)

    def compute_payoffs(self):
        return self.current_values - self.prev_values

    def update_weights(self):
        payoffs = self.compute_payoffs()
        new_weights = self.weights * jnp.exp(self.eta * payoffs)
        self.weights = new_weights / jnp.sum(new_weights)
        self.prev_values = self.current_values

    def observe_payoffs(self, values):
        """Called externally to set Vₜ₊₁(b)"""
        self.current_values = values

    @staticmethod
    def sample_action(weights, key):
        action_index = jax.random.choice(key, jnp.arange(weights.shape[0]), p=weights)
        return action_index
    
    def inventory_adjustment(self, Ht, wt_prev, pt, pt_prev):
        """Implements H_t · (w_t - w_{t-1})(p_{t-1} - p_t)"""
        delta_w = self.weights - wt_prev
        inventory = jnp.dot(Ht, delta_w)
        cash_flow = inventory * (pt - pt_prev)
        return inventory, cash_flow
    
def simulate_strategy_payoffs(mid_price_history, best_bid, best_ask, tick_size, n_ticks_in_book):
    num_actions = 8
    min_mid = jnp.min(mid_price_history)
    max_mid = jnp.max(mid_price_history)
    mid_exit = mid_price_history[-1]

    def single_action_payoff(action):
        # Action mappings
        bid_offsets = jnp.array([0, 1, 2, 3, 0, 2, 1, 4], dtype=jnp.int32)
        ask_offsets = jnp.array([0, 1, 2, 3, 2, 0, 4, 1], dtype=jnp.int32)
        bid_quants = jnp.array([1] * 8, dtype=jnp.int32)
        ask_quants = jnp.array([1] * 8, dtype=jnp.int32)

        tick_offset = tick_size * n_ticks_in_book

        bid_offset = bid_offsets[action]
        ask_offset = ask_offsets[action]
        bid_quant = bid_quants[action] * 10
        ask_quant = ask_quants[action] * 10

        bid_price = best_bid - bid_offset * tick_offset
        bid_price = jnp.maximum(bid_price, 0)

        ask_price = best_ask + ask_offset * tick_offset
        ask_price = jnp.maximum(bid_price + n_ticks_in_book * tick_size, ask_price)

        # Simulate fill
        bid_filled = bid_price >= min_mid
        ask_filled = ask_price <= max_mid

        # Compute profit if filled
        bid_profit = (mid_exit - bid_price) * bid_quant * bid_filled
        ask_profit = (ask_price - mid_exit) * ask_quant * ask_filled

        return bid_profit + ask_profit

    # Vectorize over all 8 actions
    payoffs = jax.vmap(single_action_payoff)(jnp.arange(num_actions))
    return payoffs

v_simulate_strategy_payoffs = jax.vmap(
    simulate_strategy_payoffs,
    in_axes=(0, 0, 0, None, None)  # batch over histories, bids, asks; tick params shared
)
    
# Initialize the agent for the new action space (0-7)
lrmAgent = LowRegretMetaAgent(num_strategies=8, eta=0.01, initial_volume=100)




#===========Define a train function so we can call it in a sweep===#
#  
def make_test(config):
    """Make train function. 
    Input: Config
    Output: Test function
    Train(rng): returns:
    Infos and params
    """
    #=========#
    # Get Keys
    #=========#
    rng = jax.random.key(0)
    rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)
    env_config=EnvironmentConfig(**config["ENV_CONFIG"])

    #===========#
    #Define Envs#
    #==========#
    env = MarketMakingEnv(
            env_config,
            key_reset,
            alphatradePath=config["ATFOLDER"]+"/val",
            window_index=config["WINDOW_INDEX"],
            episode_time=config["EPISODE_TIME"],
            ep_type=config["DATA_TYPE"],
        )

    env_params = dataclasses.replace(
            env.default_params,
            episode_time=config["EPISODE_TIME"],
        )
    


    def sample_actions(key, action_space, num_envs, mode):
        """Samples actions based on the specified mode."""
        if mode == "random":
            def sample_action(k):
                return action_space.sample(k)
            v_sample_action = jax.vmap(sample_action)
            keys = jax.random.split(key, num_envs)
            actions = v_sample_action(keys)
            return actions
        elif mode == "zero":
            return jnp.zeros((num_envs,), dtype=jnp.int32)  # Assuming discrete action space with action 0
        else:
            raise ValueError(f"Unknown action sampling mode: {mode}")


    def train(rng):
        """Train function
        input rng key
        output:
        return {"params": params, "info_train": info_train, "eval_info": eval_info}"""
        

        #Start count
        episode_count = 1
        

        #Reset training env
        rng, reset_key = jax.random.split(rng)
        _, env_state = env.reset(reset_key, env_params)

        #Intialise lists
        all_actions = []
        episode_reward = 0.0
        for _ in range(int(config["TOTAL_TIMESTEPS"])):   
            rng, step_key = jax.random.split(rng)                 
            mid_history = (env_state.best_asks[:, 0] + env_state.best_bids[:, 0]) / 2
            best_bid = env_state.best_bids[-1, 0]
            best_ask = env_state.best_asks[-1, 0]

            payoffs = simulate_strategy_payoffs(
                mid_history,
                best_bid,
                best_ask,
                tick_size=100,
                n_ticks_in_book=1,
            )

            lrmAgent.observe_payoffs(payoffs)
            lrmAgent.update_weights()

            # Sample action based on config
            action_mode = config["ACTION_SAMPLING_MODE"]

            if action_mode in ["random", "zero"]:
                action = sample_actions(step_key, env.action_space(), num_envs=1, mode=action_mode)[0]
            else:
                action_key, step_key = jax.random.split(step_key)
                action = lrmAgent.sample_action(lrmAgent.weights, action_key)


            if wandbOn:
                def log_action_distribution(action_val):
                    wandb.log({f"action_{int(action_val)}": 1})
                jax.debug.callback(log_action_distribution, action)

            all_actions.append(int(jax.device_get(action)))

            # Step environment
            _, env_state, reward, done, info_train = env.step(step_key, env_state, action, env_params)

            episode_reward += float(reward)

            if done:
                final_pnl = float(info_train["total_PnL"])  
                print(f"Episode done. Return: {episode_reward}, PnL: {final_pnl}")

                wandb.log(
                    data={
                        "episodic_return": episode_reward,
                        "episodic_pnl": final_pnl,
                        "episode_count": episode_count,
                    },
                    commit=True
                )

                # Reset reward tracker for next episode
                episode_reward = 0.0
                episode_count+=1

        return { "info_train": info_train}
    return train

if __name__ == "__main__":
    timestamp=datetime.datetime.now().strftime("%m-%d_%H-%M")
    try:
        ATFolder = sys.argv[1]
        print("ATFFolder:",ATFolder)
    except:
        ATFolder = "/home/duser/AlphaTrade/training_oneDay"

    
    env_config_hps = [{"observation_space":"engineered",
                            "reward_space":"spooner_scaled",
                            "inv_penalty":"none",
                            "end_fn":"unwind_ref_price",
                            "fixed_quant_value":10,
                            "reference_price_portfolio_value":"near_touch",
                            "action_space":"AvSt",
                            "inventoryPnL_lambda":0.8,
                            "asymmetrically_dampened_lambda":0.2
                            },
                            ]

    training_parameters = {
        "ENV_NAME": {"values": ["AlphaTradeMM"]},
        "ACTION_TYPE": {"values": ["pure"]},
        "TOTAL_TIMESTEPS":{"values":[5e5]},
        "WINDOW_INDEX": {"values": [-1]},
        "EPISODE_TIME": {"values": [60*5]},
        "DATA_TYPE": {"values": ["fixed_time"]},
        "ATFOLDER": {"values": [ATFolder]},
        "ENV_CONFIG": {"values": env_config_hps},
        "FLOAT_TYPE": {"values": ["float16"]},
        "ACTION_SAMPLING_MODE":{"values": ["random"]}#,"random""zero","MMMW",
    }    

    
    sweep_config={
        "method": "grid",
        "parameters": training_parameters
    }

    def sweep_fun():
            run = wandb.init(
                project="Alphatrade_Sweeps",
                save_code=True,  # 
            )
            params_file_name = f'params_file_{wandb.run.name}_{datetime.datetime.now().strftime("%m-%d_%H-%M")}'
            print(f"Results will be saved to {params_file_name}")
            # +++++ Single GPU +++++
            rng = jax.random.PRNGKey(1)
            train = (make_test(wandb.config))
            # print("+++++++++++ Training turned off whilst debugging wandb ++++++++++++")
            out = train(rng)
            run.finish()

    sweep_id = wandb.sweep(sweep=sweep_config, project="MM_BID_ASK_TEST")
    wandb.agent(sweep_id, function=sweep_fun, count=500)


    sys.exit(0)




   