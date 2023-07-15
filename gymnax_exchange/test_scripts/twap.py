# ============== testing scripts ===============
import jax
import jax.numpy as jnp
import sys
from re import L
import time 
sys.path.append('/Users/sasrey/AlphaTrade')
sys.path.append('/homes/80/kang/AlphaTrade')
import chex
import faulthandler; faulthandler.enable()
chex.assert_gpu_available(backend=None)
from jax import config # Code snippet to disable all jitting.
config.update("jax_disable_jit", False)
# config.update("jax_disable_jit", True)
from gymnax_exchange.jaxen.exec_env import *
import json
# ============== testing scripts ===============

if __name__ == "__main__":
    try:
        ATFolder = sys.argv[1]
        print("AlphaTrade folder:",ATFolder)
    except:
        ATFolder = '/homes/80/kang/AlphaTrade'
        
    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)

    env=ExecutionEnv(ATFolder,"sell")
    env_params=env.default_params
    print(env_params.message_data.shape, env_params.book_data.shape)

    start=time.time()
    obs,state=env.reset(key_reset,env_params)
    print("State after reset: \n",state)
    print("Time for reset: \n",time.time()-start)
    print(env_params.message_data.shape, env_params.book_data.shape)

    for i in range(1,100):
        # ==================== ACTION ====================
        # ---------- acion from given strategy  ----------
        test_action=env.action_space().sample(key_policy)
        
        # env_state, last_obs, last_done, rng = runner_state
        # rng, _rng = jax.random.split(rng)
        # rng_action=jax.random.split(_rng, config["NUM_ENVS"])
        # action = jax.vmap(env.action_space().sample, in_axes=(0))(rng_action)
        # rng, _rng = jax.random.split(rng)
        # rng_step = jax.random.split(_rng, config["NUM_ENVS"])
        # obsv_step, env_state_step, reward_step, done_step, info_step = jax.vmap(
        #     env.step, in_axes=(0, 0, 0, None)
        # )(rng_step, env_state, action, env_params)
        
        def twap(obs, state, params):
            # ---------- ifMarketOrder ----------
            remainingTime = params.episode_time - jnp.array((state.time-state.init_time)[0], dtype=jnp.int32)
            marketOrderTime = jnp.array(60, dtype=jnp.int32) # in seconds, means the last minute was left for market order
            ifMarketOrder = (remainingTime <= marketOrderTime)
            # ---------- ifMarketOrder ----------
            # ---------- prices ----------
            task = ppo_config["TASKSIDE"]
            tick_size = 100
            n_ticks_in_book = 20 
            best_ask, best_bid = state.best_asks[-1,0], state.best_bids[-1,0]
            A = best_bid if task=='sell' else best_ask # aggressive would be at bids
            M = (best_bid + best_ask)//2//tick_size*tick_size 
            P = best_ask if task=='sell' else best_bid
            PP= best_ask+ tick_size* n_ticks_in_book if  task=='sell' else best_bid-tick_size*n_ticks_in_book
            prices = [A,M,P,PP]
            # ---------- prices ----------
            # ---------- quants ----------
            remainedQuant = state.task_to_execute - state.quant_executed
            remainedStep = params.max_steps_in_episode - state.step_counter
            stepQunt =  remainedQuant if ifMarketOrder else remainedQuant//remainedStep
            quants = [remainedQuant//4, remainedQuant//4, remainedQuant//4, remainedQuant - 3*remainedQuant//4]
            # ---------- quants ----------
            return jnp.array([prices, quants])
            
            
        action_twap = twap(obs, state, env_params)
        
        
        
        # ---------- acion from random sampling ----------
        test_action=env.action_space().sample(key_policy)
        # ---------- acion from trained network ----------
        ac_in = (obs[np.newaxis, :], obs[np.newaxis, :])
        ## import ** network
        from gymnax_exchange.jaxrl.ppoRnnExecCont import ActorCriticRNN
        ppo_config = {
            "LR": 2.5e-4,
            "NUM_ENVS": 4,
            "NUM_STEPS": 2,
            "TOTAL_TIMESTEPS": 5e5,
            "UPDATE_EPOCHS": 4,
            "NUM_MINIBATCHES": 4,
            "GAMMA": 0.99,
            "GAE_LAMBDA": 0.95,
            "CLIP_EPS": 0.2,
            "ENT_COEF": 0.01,
            "VF_COEF": 0.5,
            "MAX_GRAD_NORM": 0.5,
            "ENV_NAME": "alphatradeExec-v0",
            "ANNEAL_LR": True,
            "DEBUG": True,
            "NORMALIZE_ENV": False,
            "ATFOLDER": ATFolder,
            "TASKSIDE":'buy'
        }
        # runner_state = np.load("runner_state.npy") # FIXME/TODO save the runner_state after training
        # network = ActorCriticRNN(env.action_space(env_params).shape[0], config=ppo_config)
        # hstate, pi, value = network.apply(runner_state.train_state.params, hstate, ac_in)
        # action = pi.sample(seed=rng) # 4*1, should be (4*4: 4actions * 4envs)
        # ==================== ACTION ====================
        
        
        print(f"Sampled {i}th actions are: ",test_action)
        start=time.time()
        obs,state,reward,done,info=env.step(key_step, state,test_action, env_params)
        print(f"State after {i} step: \n",state,done,file=open('output.txt','a'))
        print(f"Time for {i} step: \n",time.time()-start)

    # ####### Testing the vmap abilities ########
    
    # enable_vmap=True
    # if enable_vmap:
    #     vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
    #     vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))
    #     vmap_act_sample=jax.vmap(env.action_space().sample, in_axes=(0))

    #     num_envs = 10
    #     vmap_keys = jax.random.split(rng, num_envs)

    #     test_actions=vmap_act_sample(vmap_keys)
    #     print(test_actions)

    #     start=time.time()
    #     obs, state = vmap_reset(vmap_keys, env_params)
    #     print("Time for vmap reset with,",num_envs, " environments : \n",time.time()-start)

    #     start=time.time()
    #     n_obs, n_state, reward, done, _ = vmap_step(vmap_keys, state, test_actions, env_params)
    #     print("Time for vmap step with,",num_envs, " environments : \n",time.time()-start)