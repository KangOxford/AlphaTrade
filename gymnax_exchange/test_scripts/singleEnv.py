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
        # ATFolder = '/home/duser/AlphaTrade'
        # ATFolder = '/homes/80/kang/AlphaTrade'
        # ATFolder = '/homes/80/kang/AlphaTrade/testing'
        # ATFolder = '/homes/80/kang/AlphaTrade/testing_small'
        ATFolder = "/homes/80/kang/AlphaTrade/training_oneDay"
        
    env=ExecutionEnv(ATFolder,"sell")
    env_params=env.default_params
    print(env_params.message_data.shape, env_params.book_data.shape)
    assert env.task_size == 500    
    
    # -----------------------------------        
    import time
    timestamp = str(int(time.time()))
    rngInitNum = 1    
    rng = jax.random.PRNGKey(rngInitNum)
    
    rng, key_reset, key_policy, key_step, _rng = jax.random.split(rng, 5)
    obs,state=env.reset(key_reset,env_params)
    ppo_config = {
            "LR": 2.5e-4,
            "NUM_ENVS": 1,
            "NUM_STEPS": 1,
            "TOTAL_TIMESTEPS": 1e7,
            "UPDATE_EPOCHS": 1,
            "NUM_MINIBATCHES": 1,
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
            "TASKSIDE":'sell'
        }
    import flax
    from gymnax_exchange.jaxrl.ppoRnnExecCont import ActorCriticRNN
    from gymnax_exchange.jaxrl.ppoRnnExecCont import ScannedRNN
    # with open(paramsFile, 'rb') as f:
    # # with open('/homes/80/kang/AlphaTrade/params_file_prime-armadillo-72_07-17_11-02', 'rb') as f:
    #     restored_params = flax.serialization.from_bytes(flax.core.frozen_dict.FrozenDict, f.read())
    #     print(f"pramas restored")
        
    # initail_random_params = ???? # TODO
    
    network = ActorCriticRNN(env.action_space(env_params).shape[0], config=ppo_config)
    init_hstate = ScannedRNN.initialize_carry(ppo_config["NUM_ENVS"], 128)
    init_x = (
            jnp.zeros(
                (1, ppo_config["NUM_ENVS"], *env.observation_space(env_params).shape)
            ),
            jnp.zeros((1, ppo_config["NUM_ENVS"])),
        )
    _rng,_= jax.random.split(_rng, 2)
    network_params = network.init(_rng, init_hstate, init_x)
    
    
    initail_random_params=network_params
    init_done = jnp.array([False]*ppo_config["NUM_ENVS"])
    ac_in = (obs[np.newaxis, np.newaxis, :], init_done[np.newaxis, :])
    assert len(ac_in[0].shape) == 3
    hstate, pi, value = network.apply(initail_random_params, init_hstate, ac_in)
    print("Network Carry Initialized")
    action = pi.sample(seed=rng).round().astype(jnp.int32)[0,0,:].clip( 0, None) # CAUTION about the [0,0,:], only works for num_env=1
    print(f"-------------\nPPO 0th actions are: {action} with sum {action.sum()}")
    obs,state,reward,done,info=env.step(key_step, state, action, env_params)
    print("{" + ", ".join([f"'{k}': {v}" for k, v in info.items()]) + "}")
    start=time.time()
    obs,state=env.reset(key_reset,env_params)
    print("Time for reset: \n",time.time()-start)
    print(env_params.message_data.shape, env_params.book_data.shape)
    erlist = []
    excuted_list = []
    er = 0
    for i in range(1,10000):
        # ==================== ACTION ====================
        # ---------- acion from trained network ----------
        ac_in = (obs[np.newaxis,np.newaxis, :], jnp.array([done])[np.newaxis, :])
        assert len(ac_in[0].shape) == 3, f"{ac_in[0].shape}"
        assert len(ac_in[1].shape) == 2, f"{ac_in[1].shape}"
        hstate, pi, value = network.apply(network_params, hstate, ac_in) 
        action = pi.sample(seed=rng).round().astype(jnp.int32)[0,0,:].clip( 0, None)
        # ---------- acion from trained network ----------
        # ==================== ACTION ====================    
        print(f"Sampled {i}th actions are: ",action)
        start=time.time()
        obs,state,reward,done,info=env.step(key_step, state,action, env_params)
        er+=reward
        print(f"Time for {i} step: \n",time.time()-start)
        print("excuted ",info["quant_executed"])
        excuted_list.append(info["quant_executed"])
        if done:
            erlist.append((i, er))
            print(f"windowIndex {:<10} , global step {i:<10} , episodic return {er:^20} , ",\
                file=open('twap_'+ timestamp +"_OneDay_train_"+'.txt','a'))
            network = ActorCriticRNN(env.action_space(env_params).shape[0], config=ppo_config)
            init_hstate = ScannedRNN.initialize_carry(ppo_config["NUM_ENVS"], 128)
            init_x = (
                    jnp.zeros(
                        (1, ppo_config["NUM_ENVS"], *env.observation_space(env_params).shape)
                    ),
                    jnp.zeros((1, ppo_config["NUM_ENVS"])),
                )
            _rng,_= jax.random.split(_rng,2)
            network_params = network.init(_rng, init_hstate, init_x)
            key_reset, _ = jax.random.split(key_reset,2)
            obs,state=env.reset(key_reset,env_params)
            er =0
            