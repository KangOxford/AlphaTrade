# docker run -it --rm --gpus '"device=7"' -v $(pwd):/app -v $(pwd)/../cache:/app/cache --name ${USER}_lcc ${USER}_lc python -m scripts.rl_test

# tokens are 0: pad, 1, 2: actions, 3 -> 258: observations
import jax
import sys
import os
sys.path.append(os.path.abspath('/home/duser/AlphaTrade')) 
import jax.numpy as jnp
#import flax.linen as nn
import datetime
import numpy as np
import optax
import time
from dataclasses import dataclass
import pickle

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_enable_triton_softmax_fusion=true '
    '--xla_gpu_triton_gemm_any=True '
    # '--xla_gpu_enable_async_collectives=true '
    # '--xla_gpu_enable_latency_hiding_scheduler=true '
    # '--xla_gpu_enable_highest_priority_async_stream=true '
)

from typing import Sequence, NamedTuple, Any, Dict, Callable, Optional
from transformers import PreTrainedTokenizerFast

import distrax
import gymnax
import functools
from gymnax.environments import spaces
from gymnax_exchange.jaxrl.utils import FlattenObservationWrapper, LogWrapper
from jax._src import dtypes
from gymnax_exchange.jaxen.mm_env import MarketMakingEnv 
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
import pandas as pd

from jax_rwkv.src.auto import get_rand_model
from gymnax_exchange.jaxrl.rl_processing import get_ppo_agent, calculate_gae, get_jit_ppo, PAD_FLAG, OBS_FLAG, ACT_FLAG
#from utils.jstring import JString

j_calculate_gae = jax.jit(jax.vmap(calculate_gae, in_axes=(0, 0, 0, 0, 0, None, None)))

import wandb


wandbOn =  False
if wandbOn:
    import wandb

@jax.tree_util.register_pytree_node_class
@dataclass
class JString:
    tokens: jnp.ndarray
    length: jnp.ndarray

    def __init__(self, tokens, length=None):
        self.tokens = jnp.array(tokens)
        self.length = (
            length if length is not None
            else compute_true_length(tokens, pad_token=pad_token_id)
        )

    def tree_flatten(self):
        # The children are the arrays that JAX can trace.
        children = (self.tokens, self.length)
        # No auxiliary static data.
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        tokens, length = children
        return cls(tokens, length)

##Get the AT folder
try:
        ATFolder = sys.argv[1]
        print("AlphaTrade folder:",ATFolder)
except:
        # ATFolder = "./testing_oneDay"
        #ATFolder = "/training_oneDay"
        ATFolder = "/home/duser/AlphaTrade/training_oneDay"

config = {
    "LR": 1e-3,
    "NUM_ENVS": 2,
    "NUM_STEPS": 10,#128,
    "TOTAL_TIMESTEPS": 4e5,
    "UPDATE_EPOCHS": 4,
    "NUM_MINIBATCHES": 1,
    "GAMMA": 0.99,# ** (1/5),
    "GAE_LAMBDA": 0.95 ,#** (1/5),
    "CLIP_EPS": 0.2,
    "ENT_COEF": 0.1,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "ACTIVATION": "relu",
    "ANNEAL_LR": False,
    "DEBUG": True,
    "WANDB": True,

     "TASKSIDE": "random", # "random", "buy", "sell"
        "REWARD_LAMBDA": 0.2, #0.001,
        "ACTION_TYPE": "pure", # "delta"
        "WINDOW_INDEX": 43, # 2 fix random episode #-1,
        "EPISODE_TIME": 60*8,  # 
        "DATA_TYPE": "fixed_time", # "fixed_time", "fixed_steps"
        "ATFOLDER": ATFolder,
        "MAX_SEQ_LEN":8000
    }


config["NUM_UPDATES"] = (
    config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
)


jit_ppo_update = get_jit_ppo(config)



rng = jax.random.key(0)
rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)

env = MarketMakingEnv(
        key_reset,
        alphatradePath=config["ATFOLDER"],
        window_index=config["WINDOW_INDEX"],
        episode_time=config["EPISODE_TIME"],
        ep_type=config["DATA_TYPE"],
    )

env_params = dataclasses.replace(
        env.default_params,
        reward_lambda=config["REWARD_LAMBDA"],
        episode_time=config["EPISODE_TIME"],
    )
if wandbOn:
    run = wandb.init(
        project="AlphaTradeJAX_rwkv_Train",
        config=config,
        save_code=True,
        reinit=True  # Ensures logging works even if script restarts
    )

if wandbOn:
    def log_all_metrics(info,global_timestep):
        """Logs training metrics to wandb in real-time using jax.debug.callback."""
        return_values = info["returned_episode_returns"][info["returned_episode"]]
        timesteps = info["timestep"][info["returned_episode"]] * config["NUM_ENVS"]
        PnL = info["total_PnL"]
        inventories = info["inventory"]
        buyQuant = info["buyQuant"]
        sellQuant = info["sellQuant"]
        reward = info["reward"]
        netWorth = info["netWorth"]
        other_exec_quants = info["other_exec_quants"]
        inventoryValue=info["inventoryValue"]

        # Extract the last PnL per finished episode for all environments
        final_PnL_per_env = PnL[info["returned_episode"]] if PnL.size > 0 and info["returned_episode"].size > 0 else jnp.array([])

        # Compute the average final PnL across environments
        avg_final_PnL = jnp.mean(final_PnL_per_env) if final_PnL_per_env.size > 0 else 0

        # Log all existing metrics + final PnL
        wandb.log(
            {
                "global_step": jnp.max(timesteps) if timesteps.size > 0 else 0,
                "reward": jnp.mean(reward) if reward.size > 0 else 0,
                "episodic_return": jnp.mean(return_values) if return_values.size > 0 else 0,
                "PnL": jnp.mean(PnL) if PnL.size > 0 else 0,
                "inventory": jnp.mean(inventories) if inventories.size > 0 else 0,
                "buyQuant": jnp.mean(buyQuant) if buyQuant.size > 0 else 0,
                "sellQuant": jnp.mean(sellQuant) if sellQuant.size > 0 else 0,
                "other_exec_quants": jnp.mean(other_exec_quants) if other_exec_quants.size > 0 else 0,
                "avg_final_PnL": avg_final_PnL,  # NEW: Log average final PnL across envs
                "netWorth":jnp.mean(netWorth)if netWorth.size>0 else 0,
                "inventoryValue":jnp.mean(inventoryValue) if inventoryValue.size>0 else 0
            },
            commit=True,  # Ensures immediate update in wandb
        )

# Load the tokenizer
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="gymnax_exchange/jaxlobster/lob_tok.json",
    clean_up_tokenization_spaces=False
)

import jax.numpy as jnp

import jax.numpy as jnp

def lob_to_str(jax_array, n_msgs=100):
    # Constants to match the target format
    TIME_COL = "<time>"
    EVENT_TYPE_COL = "<event_type>"
    ORDER_ID_COL = "<order_id>"
    SIZE_COL = "<size>"
    PRICE_COL = "<price>"
    DIRECTION_COL = "<direction>"

    # Extract relevant columns from the JAX array
    T = jax_array[:, 0]
    S = jax_array[:, 1]
    Q = jax_array[:, 2]
    P = jax_array[:, 3]
    OID = jax_array[:, 4]
    TID = jax_array[:, 5]
    Ts = jax_array[:, 6]
    Tns = jax_array[:, 7]

    # Adjust nanoseconds to seconds (without formatting)
    time_col = Ts + (Tns / 1e9)

    # Map event type from the T column
    event_type_col = jnp.select([T == 1, T == 2, T == 3], [1, 3, 2], default=0)

    # Convert to string outside JAX computation
    time_col_str = [f"{x:.9f}" for x in time_col]  

    rows = []
    for i in range(jax_array.shape[0]):
        row = [
            (TIME_COL, time_col_str[i]),
            (EVENT_TYPE_COL, int(event_type_col[i])),
            (ORDER_ID_COL, int(OID[i])),
            (SIZE_COL, int(Q[i])),
            (PRICE_COL, int(P[i])),
            (DIRECTION_COL, int(S[i])),
        ]
        row_str = ','.join([f"{col},{val}" for col, val in row])
        rows.append(row_str)

    # Batch the rows into messages
    batched_strings = [
        '\n'.join(rows[i:i + n_msgs])
        for i in range(0, len(rows), n_msgs)
    ]
    batched_strings = "\n".join(batched_strings)

    return batched_strings



pad_token_id = 3
print("Padding token ID:", pad_token_id)


def tokenize_observation(observation):
    observation_str = [str(obs) for obs in observation]  
    return jnp.array([tokenizer.encode(obs) for obs in observation_str], dtype=jnp.int32)

def pad_to_max_length(tokens, max_len, pad_token=0):
    padded_tokens = jnp.array([
        jnp.pad(t, (0, max_len - len(t)), constant_values=pad_token)
        for t in tokens
    ])
    return padded_tokens


def compute_true_length(tokens, pad_token=3):
    # Calculate the true lengths and expand dimensions to (num_envs, 1
    return jnp.expand_dims(jnp.sum(tokens != pad_token, axis=1), axis=1)

def remove_padding(tokens, pad_token_id, token_lengths):
    return tokens[:, :token_lengths.max()]  # Select only valid tokens (not padded)

# Load the pretrained model
with open("gymnax_exchange/jaxrl/pre_trained_weights/goog2022_rwkv_6g0.1B.model", "rb") as f:
    pretrained_params = pickle.load(f)


# Add 8 new action tokens
new_actions = [f"<action_{i}>" for i in range(env.action_space(env_params).n)] 
tokenizer.add_special_tokens({"additional_special_tokens": new_actions})

# Update action token range in config
config["MIN_ACTION_TOK"] = tokenizer.convert_tokens_to_ids("<action_0>")
config["MAX_ACTION_TOK"] = tokenizer.convert_tokens_to_ids("<action_7>")

# Get updated vocab size (considering action tokens)
#potentially remove 1
num_tokens =  env.action_space(env_params).n + tokenizer.vocab_size

print(f"Updated vocab size: {num_tokens}")

old_vocab_size = pretrained_params['emb']['weight'].shape[0]
n_embd =  pretrained_params['emb']['weight'].shape[1]

# Expand embeending for the new tokens
if num_tokens > old_vocab_size:
    pad_shape = (num_tokens - old_vocab_size, n_embd)
    new_embeddings = jnp.zeros(pad_shape, dtype=pretrained_params['emb']['weight'].dtype)

    # Concatenate new embeddings
    pretrained_params['emb']['weight'] = jnp.concatenate(
        [pretrained_params['emb']['weight'], new_embeddings], axis=0
    )

print("Embedding layer updated to size:", pretrained_params['emb']['weight'].shape)

# Get N_layers
n_layer =  pretrained_params['blocks']['att']['time_faaaa'].shape[0]
print("nlayer:",n_layer)

# Wrap the environment
env = FlattenObservationWrapper(env)
env = LogWrapper(env)


# Initialize the RWKV model with dynamic layer and embedding size
RWKV, _ = get_rand_model(0, "6", n_layer, n_embd, num_tokens, dtype="float32", rwkv_type="ScanRWKV")
params = pretrained_params 

print("Original head layer shape:", pretrained_params['head']['weight'].shape)


forward, params = get_ppo_agent(RWKV, params, seed=1)
v_forward_jit = jax.jit(jax.vmap(forward, in_axes=(0, 0, None, 0)))
init_state = RWKV.default_state(params)
#if isinstance(init_state, tuple):
#    init_state = tuple([
#        jnp.repeat(s[None], config["NUM_ENVS"], axis=0) if len(s.shape) == 1 else s for s in init_state
#    ])
    #init_state = tuple([jnp.repeat(s[None], config["NUM_ENVS"], axis=0) for s in init_state])
#else:
#    init_state = jnp.repeat(init_state[None], config["NUM_ENVS"], axis=0)

if isinstance(init_state, tuple):
    # If it's a tuple, check each element to ensure correct batching
    # Repeat each element of the tuple (if they have the correct dimension) along the batch dimension (NUM_ENVS)
    batched_state = tuple([
        jnp.repeat(s[None], config["NUM_ENVS"], axis=0) if len(s.shape) == 1 else s
        for s in init_state
    ])
else:
    # If it's not a tuple, just repeat along the batch dimension (NUM_ENVS)
    batched_state = jnp.repeat(init_state[None], config["NUM_ENVS"], axis=0)

# Now batched_state should have the correct shape
state = batched_state

#state = init_state
jax.debug.print("state shape: {}", state.shape)

def linear_schedule(count):
    frac = (
        1.0
        - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
        / config["NUM_UPDATES"]
    )
    return config["LR"] * frac

solver = optax.chain(
    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
    optax.adam(linear_schedule, eps=1e-5)
)
optimizer = solver.init(params)

rng, _rng = jax.random.split(rng)
reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

v_env_step = jax.jit(jax.vmap(
    env.step, in_axes=(0, 0, 0, None)
))

global_timestep = 1

for _ in range(int(config["TOTAL_TIMESTEPS"]) // config["NUM_STEPS"] // config["NUM_ENVS"]):
    initial_state = state
    tokens_list = []
    flags_list = []
    values_list = []
    rewards_list = []
    log_prob_list = []
    dones_list = []

    all_actions = []
    update_returns = []
    for t in range(config["NUM_STEPS"]):
        rng, _rng = jax.random.split(rng)
        jax.debug.print("obsv shape{}:", obsv.shape[1])

        # Reshape the observation for tokenization
        reshaped_obsv = obsv.reshape(config["NUM_ENVS"], obsv.shape[1] // 8, 8)

        # Collect tokenized sequences as a Python list
        tokenized_list = []

        for env_obs in reshaped_obsv:
            obs_str = lob_to_str(env_obs)
            tokenized = tokenizer.encode(obs_str)
            tokenized_list.append(tokenized)

        # Determine the lengths before padding
        token_lengths = jnp.array([len(t) for t in tokenized_list])

        # Pad all tokenized sequences to MAX_SEQ_LEN and convert to jnp array
        tokenized_observations = jnp.array([
            jnp.pad(jnp.array(t), (0, config["MAX_SEQ_LEN"] - len(t)), mode='constant', constant_values=pad_token_id)
            for t in tokenized_list
        ])
        print("tokenized obs shape 0: {}",tokenized_observations.shape[0])
        print("tokenized obs shape 1: {}",tokenized_observations.shape[1])

        # Compute true lengths in (num_envs, 1) shape
        token_lengths = compute_true_length(tokenized_observations, pad_token=pad_token_id)
        print(token_lengths)
        jax.debug.print("state shape: {}", state.shape)
        jax.debug.print("state[0] shape: {}", state[0].shape)  
        jax.debug.print("state[1] shape: {}", state[1].shape)  

        # Forward pass
        pi, value, state = v_forward_jit(tokenized_observations, state, params, token_lengths)

        pi = distrax.Categorical(logits=pi[..., -1, config["MIN_ACTION_TOK"]:config["MAX_ACTION_TOK"] + 1])
        action = pi.sample(seed=_rng)
        jax.debug.print("action:{}", action)

        def log_action_distribution(action):
            unique_actions, counts = jnp.unique(action, return_counts=True)
            action_distribution = {f"action_{int(a)}": int(c) for a, c in zip(unique_actions, counts)}
            wandb.log(action_distribution)

        if wandbOn:
            jax.debug.callback(log_action_distribution, action)

        # Store actions
        current_actions = jax.device_get(action)
        all_actions.extend(current_actions.flatten().tolist())

        # Compute log probability of the action
        log_prob = pi.log_prob(action)
        _, value1, state = v_forward_jit(action, state, params, jnp.ones(config["NUM_ENVS"], dtype=jnp.int32))

        # Step the environment
        rng, _rng = jax.random.split(rng)
        rng_step = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state, reward, done, info = v_env_step(rng_step, env_state, action, env_params)

        if wandbOn:
            jax.debug.callback(log_all_metrics, info, global_timestep)

        # Reset state for done episodes
        state = jax.vmap(jax.lax.select)(done, init_state, state)

        # Create flags for the observation
        obs_flags = jnp.ones_like(tokenized) * OBS_FLAG  # Default to OBS_FLAG
        obs_flags = jnp.where(tokenized == pad_token_id, PAD_FLAG, obs_flags)  # Flag padding
        flags_list.append(obs_flags)
        tokens_list.append(tokenized)
        
        # Append action tokens and their flags
        action_tokens = action[:, None] + config["MIN_ACTION_TOK"]
        tokens_list.append(action_tokens)
        flags_list.append(jnp.ones_like(action_tokens) * ACT_FLAG)  

        values_list.append(value)
        values_list.append(value1)

        
        obs_rewards = jnp.zeros_like(tokenized)
        rewards_list.append(obs_rewards)
        rewards_list.append(reward[:, None]) 

    
        obs_log_probs = jnp.zeros_like(value)
        log_prob_list.append(obs_log_probs)
        log_prob_list.append(log_prob[:, None]) 

    
        obs_dones = jnp.zeros_like(value, dtype=jnp.bool)
        dones_list.append(obs_dones)
        dones_list.append(done[:, None])  # Actions have their own dones

        # Track episode returns
        return_values = info["returned_episode_returns"][info["returned_episode"]]
        for r in return_values:
            update_returns.append(r)
        global_timestep += 1

    # Concatenate all lists
    tokens_list = jnp.concatenate(tokens_list, axis=1)
    flags_list = jnp.concatenate(flags_list, axis=1)
    values_list = jnp.concatenate(values_list, axis=1)
    rewards_list = jnp.concatenate(rewards_list, axis=1)
    log_probs_list = jnp.concatenate(log_prob_list, axis=1)[..., 1:]
    dones_list = jnp.concatenate(dones_list, axis=1)
    true_lengths = compute_true_length(tokens_list, pad_token=pad_token_id)
    buf = JString(tokens_list, true_lengths)

    # Mark padding as done to avoid bootstrapping
    dones_list = jnp.cumsum(dones_list, axis=1, dtype=jnp.bool)
    flags_list = jnp.where(jnp.concatenate((dones_list[:, :1], dones_list[:, :-1]), axis=1), PAD_FLAG, flags_list)

    # Tokenize the final observation
    tokenized_obsv = tokenize_observation(obsv)
    tokenized_obsv = pad_to_max_length(tokenized_obsv, config["MAX_SEQ_LEN"], pad_token_id)
        # Compute true lengths for the final observation
    final_token_lengths = compute_true_length(tokenized_obsv, pad_token=pad_token_id)

    # Compute the last value for GAE
    _, last_value, _ = v_forward_jit(tokenized_obsv, state, params, final_token_lengths)


    # Compute advantages and targets with masking
    mask = (flags_list != PAD_FLAG)  # Mask out padding

    advantages, targets = j_calculate_gae(flags_list,dones_list,values_list,rewards_list* mask,last_value,config["GAMMA"],config["GAE_LAMBDA"])
    # Log average returns
    if len(update_returns) > 0:
        print("avg returns:", sum(update_returns) / len(update_returns))
    else:
        print("None ended")

    # Update parameters using PPO
    for _ in range(config["UPDATE_EPOCHS"]):
        params, optimizer, (loss, value_loss, loss_actor, entropy, state) = jit_ppo_update(
            solver, v_forward_jit, params, optimizer, buf, flags_list, values_list, log_probs_list, advantages, targets, initial_state
        )
        print(loss, value_loss, loss_actor, entropy)

    # Reset state for done episodes
    state = jax.vmap(jax.lax.select)(dones_list[:, -1], init_state, state)