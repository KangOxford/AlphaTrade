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
import distrax
from jax_rwkv.src.auto import get_rand_model

from gymnax_exchange.jaxrl.rl_processing import get_ppo_agent, calculate_gae, get_jit_ppo, PAD_FLAG, OBS_FLAG, ACT_FLAG
from jax import lax


wandbOn = True # False
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

def tokenize_observation(observation):
    observation_str = [lob_to_str(obs) for obs in observation]  
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

def remove_padding(obvs_tokenized, obvs_token_lengths):
    # Use jax.vmap to apply the slicing over all environments
    return jax.vmap(lambda tokens, length: tokens[:length])(obvs_tokenized, obvs_token_lengths)


def lob_to_str(jax_array, n_msgs=10):
        # Constants to match the target format
        TIME_COL = "<time>"
        EVENT_TYPE_COL = "<event_type>"
        ORDER_ID_COL = "<order_id>"
        TRADER_ID_COL = "<trader_id>"
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
                (EVENT_TYPE_COL, int(event_type_col[i])),
                (DIRECTION_COL, int(S[i])),
                (SIZE_COL, int(Q[i])),
                (PRICE_COL, int(P[i])),
                (ORDER_ID_COL, int(OID[i])),
                (TRADER_ID_COL, int(TID[i])),
                (TIME_COL, time_col_str[i]),                
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
def update_flags_for_env(flags_array, length):
    return lax.dynamic_update_slice(flags_array, jnp.ones((length,), dtype=flags_array.dtype) * OBS_FLAG, (0, 0))
def update_flags_across_envs(obvs_token_lengths, obvs_flags):
    return jax.vmap(update_flags_for_env, in_axes=(None, 0))(obvs_flags, obvs_token_lengths)

if __name__ == "__main__":
    timestamp=datetime.datetime.now().strftime("%m-%d_%H-%M")
    try:
        ATFolder = sys.argv[1]
        print("ATFFolder:",ATFolder)
    except:
        ATFolder = "/home/duser/AlphaTrade/training_oneDay"
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
        "NUM_ENVS": 4,
        "NUM_STEPS": 1,#10,#128,
        "TOTAL_TIMESTEPS": 100000,#4e5,
        "UPDATE_EPOCHS": 2,
        "NUM_MINIBATCHES": 4,
        "GAMMA": 0.99,# ** (1/5),
        "GAE_LAMBDA": 0.95 ,#** (1/5),
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.1,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "relu",
        "ANNEAL_LR": False,
        "DEBUG": True,
        "EPISODE_TIME":60*5,
        "WINDOW_INDEX": 20, # 2 fix random episode #-1,
        "DATA_TYPE": "fixed_time", # "fixed_time", "fixed_steps"
        "ATFOLDER": ATFolder,
        "MAX_SEQ_LEN":4096
        }
    
    if wandbOn:
        run = wandb.init(
            project="AlphaTradeMessageInputTests",
            config=config,
            save_code=True,  # optional
        )


    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )


 
    jit_ppo_update = get_jit_ppo(config)



    rng = jax.random.key(0)
    rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)

    env_config=EnvironmentConfig(observation_space="messages",
                                 reward_space="portfolio_value",
                                 inv_penalty="linear",
                                 end_fn="unwind_ref_price",
                                 fixed_quant_value=10,
                                 reference_price_portfolio_value="mid",
                                 action_space="fixed_quants")
    env = MarketMakingEnv(
                env_config,
                key_reset,
                alphatradePath=config["ATFOLDER"]+"/train",
                window_index=config["WINDOW_INDEX"],
                episode_time=config["EPISODE_TIME"],
                ep_type=config["DATA_TYPE"],
            )
    env_params = dataclasses.replace(
                env.default_params,
                episode_time=config["EPISODE_TIME"],
            )
        

    pad_token_id=3

    # Load the tokenizer
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file="gymnax_exchange/jaxlobster/lob_tok.json",
        clean_up_tokenization_spaces=False
    )

    # Load the pretrained model
    with open("gymnax_exchange/jaxrl/pre_trained_weights/goog2022_rwkv_6g0.1B.model", "rb") as f:
        pretrained_params = pickle.load(f)

    #Add new action tokens to our 10k tokenizer
    num_actions = env.action_space(env_params).n
    new_actions = [f"<action_{i}>" for i in range(num_actions)]
    tokenizer.add_special_tokens({"additional_special_tokens": new_actions})


    # Update action token range in config
    min_action_tok = tokenizer.convert_tokens_to_ids("<action_0>")
    max_action_tok = tokenizer.convert_tokens_to_ids(f"<action_{num_actions - 1}>")
    config["MIN_ACTION_TOK"] = min_action_tok
    config["MAX_ACTION_TOK"] = max_action_tok
    jax.debug.print("min_action_tok: {}", min_action_tok)
    jax.debug.print("max_action_tok: {}", max_action_tok)

    num_tokens =  num_actions + tokenizer.vocab_size

    #Intialise new embeedings
    n_embd =  pretrained_params['emb']['weight'].shape[1]
    new_action_embeddings = jnp.zeros((num_actions, n_embd), dtype=pretrained_params['emb']['weight'].dtype)

    #Put these intialised embeddings into the model
    pretrained_params['emb']['weight'] = pretrained_params['emb']['weight'].at[min_action_tok:max_action_tok+1].set(new_action_embeddings)#+1 as slicing needs to be inclusive
    
    
    # Get N_layers
    n_layer =  pretrained_params['blocks']['att']['time_faaaa'].shape[0]
    print("nlayer:",n_layer)

    # Wrap the environment
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)

    # Initialize the RWKV model with dynamic layer and embedding size
    #use the load functionality to get the model, hwoever we dont load these random weights, just using the interface to get rwkv6
    RWKV, _ = get_rand_model(0, "6", n_layer, n_embd, num_tokens, dtype="float32", rwkv_type="ScanRWKV")
    params = pretrained_params 

    # Define a forward fn
    forward, params = get_ppo_agent(RWKV, params, seed=1)
    #jit version
    v_forward_jit = jax.jit(jax.vmap(forward, in_axes=(0, 0, None, 0)))

    #Jitted GAE from rl_processing
    j_calculate_gae = jax.jit(jax.vmap(calculate_gae, in_axes=(0, 0, 0, 0, 0, None, None)))

    #Get intial state for RWKV
    init_state = RWKV.default_state(params)
    if isinstance(init_state, tuple):
        init_state = tuple([
            jnp.repeat(s[None], config["NUM_ENVS"], axis=0) if len(s.shape) == 1 else s for s in init_state
        ])
        init_state = tuple([jnp.repeat(s[None], config["NUM_ENVS"], axis=0) for s in init_state])
    else:
        init_state = jnp.repeat(init_state[None], config["NUM_ENVS"], axis=0)

    state = init_state
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
    

    #Reset the environment
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
    obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

    #Define a Jitted step function
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
            obvs_list = []
            obvs_token_lengths = []
            obsv = obsv.reshape((config["NUM_ENVS"], 104, 8))

            for i in range(obsv.shape[0]):  # loop over envs
                #non jitted np pre proc....
                env_obsv = obsv[i]  # shape (104, 8)
              #  jax.debug.print("env_obsv: {}", env_obsv)

                # Convert to numpy for Python-side processing
                env_obsv_np = jax.device_get(env_obsv)

                # String formatter
                string_obs = lob_to_str(env_obsv_np)  # string output
             #   jax.debug.print("string_obs: {}", string_obs)
                
                # Tokenize
                tokens = tokenizer.encode(string_obs)
                obvs_token_lengths.append(min(len(tokens), config["MAX_SEQ_LEN"]))
            #    jax.debug.print("obvs_token_lengths: {}", obvs_token_lengths)

                padded = tokens[:config["MAX_SEQ_LEN"]]
                padded += [pad_token_id] * (config["MAX_SEQ_LEN"] - len(padded))
        #        jax.debug.print("padded: {}", padded) 

                obvs_list.append(jnp.array(padded, dtype=jnp.int32))

            obvs_tokenized = jnp.stack(obvs_list)
         #   jax.debug.print("tokenized_batch: {}", obvs_tokenized)
          #  jax.debug.print("tokenized_batch shape: {}", obvs_tokenized.shape)

            obvs_token_lengths = jnp.array(obvs_token_lengths, dtype=jnp.int32)

            #feed in
           # jax.debug.print("obvs_list shape: {}", obvs_token_lengths.shape)  # (NUM_ENVS, MAX_SEQ_LEN)
           # jax.debug.print("token_lengths: {}", obvs_token_lengths)  # (NUM_ENVS,)

            pi, value, state = v_forward_jit(obvs_tokenized, state, params, obvs_token_lengths)
            value=value.astype(jnp.float32)
            

            pi = distrax.Categorical(logits=pi[..., -1, config["MIN_ACTION_TOK"]:config["MAX_ACTION_TOK"] + 1])
            action = pi.sample(seed=_rng)
          #  jax.debug.print("action: {}", action)


            current_actions = jax.device_get(action)
            all_actions.extend(current_actions.flatten().tolist())
            log_prob = pi.log_prob(action)
            
            _, value1, state = v_forward_jit(action, state, params, jnp.ones(config["NUM_ENVS"], dtype=jnp.int32))
            value1=value1.astype(jnp.float32)

            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, config["NUM_ENVS"])
            obsv, env_state, reward, done, info = v_env_step(rng_step, env_state, action, env_params)

            state = jax.vmap(jax.lax.select)(done, init_state, state)

           
            #jax.debug.print("obvs_tokenized shape: {}", obvs_tokenized.shape)  
            
            tokens_list.append(obvs_tokenized)#add the obvs
            tokens_list.append(action[:, None] + config["MIN_ACTION_TOK"])#add the action token
            padding_length = config["MAX_SEQ_LEN"] - obvs_token_lengths#work out length of padding for each env
        #    jax.debug.print("padding_length: {}", padding_length)  # (NUM_ENVS,)

           
            padding_mask = jnp.arange(config["MAX_SEQ_LEN"]) >= obvs_token_lengths[:, None] #pad from obvs end to max seq length
          #  jax.debug.print("padding_mask: {}", padding_mask)  
            obs_flags = jnp.ones_like(obvs_tokenized) * OBS_FLAG
            obs_pad_flags = jnp.where(padding_mask, PAD_FLAG, OBS_FLAG)


            
            flags_list.append(obs_pad_flags)#put the obvs and pad flags on
            flags_list.append(jnp.ones_like(obvs_tokenized)[:, :1] * ACT_FLAG)#put the act flag on

            values_list.append(value)
            values_list.append(value1)

            rewards_list.append(jnp.zeros(shape=obvs_tokenized.shape))#no reward for all obvs
            rewards_list.append(reward[:, None])#reward for action at end

            log_prob_list.append(jnp.zeros_like(value))
            log_prob_list.append(log_prob[:, None])

            dones_list.append(jnp.zeros(value.shape, dtype=jnp.bool))
            dones_list.append(done[:, None])

            return_values = info["returned_episode_returns"][info["returned_episode"]]
            for r in return_values:
                update_returns.append(r)
            global_timestep += 1

            if config.get("DEBUG"):

                def callback(return_values):
                    wandb.log(
                        {"episodic_return": jnp.mean(return_values) if return_values.size > 0 else 0,
                        }
                    )

        #Form lists for adv calcs
        tokens_list = jnp.concatenate(tokens_list, axis=1)
       # jax.debug.print("tokens_list shape: {}", tokens_list.shape) #(NUM_ENVS, MAX_SEQ_LEN+1 for action *num steps) 
        flags_list = jnp.concatenate(flags_list, axis=1)
        #jax.debug.print("flags_list shape: {}", flags_list.shape) #(NUM_ENVS, MAX_SEQ_LEN+1 for action *num steps)
        values_list = jnp.concatenate(values_list, axis=1)
        rewards_list = jnp.concatenate(rewards_list, axis=1)
        log_probs_list = jnp.concatenate(log_prob_list, axis=1)[..., 1:]
        dones_list = jnp.concatenate(dones_list, axis=1)
        buf = JString(tokens_list, jnp.ones_like(tokens_list[:, 0]) * tokens_list.shape[1])
      
        #some flag masking
        dones_list = jnp.cumsum(dones_list, axis=1, dtype=jnp.bool)
        flags_list = jnp.where(jnp.concatenate((dones_list[:, :1], dones_list[:, :-1]), axis=1), PAD_FLAG, flags_list)
        
        final_obvs_list = []
        final_obvs_token_lengths = []
        final_obsv = obsv.reshape((config["NUM_ENVS"], 104, 8))
        for i in range(final_obsv.shape[0]):  # loop over envs
            env_obsv = final_obsv[i]  # shape (104, 8)
            # Convert to numpy for Python-side processing
            env_obsv_np = jax.device_get(env_obsv)

            # Exisiting string formatter
            string_obs = lob_to_str(env_obsv_np)  # string output
         #   jax.debug.print("string_obs: {}", string_obs)
            
            # Tokenize
            tokens = tokenizer.encode(string_obs)
            final_obvs_token_lengths.append(min(len(tokens), config["MAX_SEQ_LEN"]))
         #   jax.debug.print("obvs_token_lengths: {}", final_obvs_token_lengths)

            padded = tokens[:config["MAX_SEQ_LEN"]]
            padded += [pad_token_id] * (config["MAX_SEQ_LEN"] - len(padded))
        #    jax.debug.print("padded: {}", padded)

            final_obvs_list.append(jnp.array(padded, dtype=jnp.int32))
        final_obvs_tokenized = jnp.stack(final_obvs_list)
        final_obvs_token_lengths = jnp.array(final_obvs_token_lengths, dtype=jnp.int32)

        # Get last value from state
        _, last_value, _ = v_forward_jit(final_obvs_tokenized, state, params, final_obvs_token_lengths)
        last_value=last_value.astype(jnp.float32)
        advantages, targets = j_calculate_gae(flags_list, dones_list, values_list, rewards_list, last_value[..., -1], config["GAMMA"], config["GAE_LAMBDA"])
        # print("value", values_list)
        # print("target", targets)
        print("UPDATING")
        if len(update_returns) > 0:
            print("avg returns:", sum(update_returns) / len(update_returns))
        else:
            print("None ended")
        
        #Update weights
        for _ in range(config["UPDATE_EPOCHS"]):
            params, optimizer, (loss, value_loss, loss_actor, entropy, state) = jit_ppo_update(solver, v_forward_jit, params, optimizer, buf, flags_list, values_list, log_probs_list, advantages, targets, initial_state)
            print(loss, value_loss, loss_actor, entropy)
        #Reset state if done (rwkv state)
        state = jax.vmap(jax.lax.select)(dones_list[:, -1], init_state, state)

