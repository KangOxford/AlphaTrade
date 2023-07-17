# from jax import config
# config.update("jax_enable_x64",True)
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
        ATFolder = '/homes/80/kang/AlphaTrade/testing_small'
        
    rngInitNum = 0
    def get_ppo_average_price(rngInitNum):
        rng = jax.random.PRNGKey(rngInitNum)
        rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)
        env=ExecutionEnv(ATFolder,"sell")
        env_params=env.default_params
        obs,state=env.reset(key_reset,env_params)
        ppo_config = {
                "LR": 2.5e-4,
                "NUM_ENVS": 1,
                "NUM_STEPS": 1,
                "TOTAL_TIMESTEPS": 5e5,
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
                "TASKSIDE":'buy'
            }
        import flax
        from gymnax_exchange.jaxrl.ppoRnnExecCont import ActorCriticRNN
        from gymnax_exchange.jaxrl.ppoRnnExecCont import ScannedRNN
        with open('/homes/80/kang/AlphaTrade/params_file_prime-armadillo-72_07-17_11-02', 'rb') as f:
        # with open('/homes/80/kang/AlphaTrade/params_file_firm-fire-68_07-17_09-53', 'rb') as f:
        # with open('/homes/80/kang/AlphaTrade/params_file_2023-07-10_12-34-24', 'rb') as f:
        # with open('/homes/80/kang/AlphaTrade/params_file_2023-07-08_15-22-20', 'rb') as f:
            restored_params = flax.serialization.from_bytes(flax.core.frozen_dict.FrozenDict, f.read())
            print(f"pramas restored")
        network = ActorCriticRNN(env.action_space(env_params).shape[0], config=ppo_config)
        init_hstate = ScannedRNN.initialize_carry(ppo_config["NUM_ENVS"], 128)
        init_done = jnp.array([False]*ppo_config["NUM_ENVS"])
        ac_in = (obs[np.newaxis, np.newaxis, :], init_done[np.newaxis, :])
        assert len(ac_in[0].shape) == 3
        hstate, pi, value = network.apply(restored_params, init_hstate, ac_in)
        print("Network Carry Initialized")
        action = pi.sample(seed=rng).round().astype(jnp.int32)[0,0,:].clip( 0, None) # CAUTION about the [0,0,:], only works for num_env=1
        print(f"-------------\nPPO 0th actions are: {action} with sum {action.sum()}")
        obs,state,reward,done,info=env.step(key_step, state, action, env_params)
        # done, state.quant_executed
        print("{" + ", ".join([f"'{k}': {v}" for k, v in info.items()]) + "}")

        
        i = 1
        for i in range(1,10000):
            # ==================== ACTION ====================
            # ---------- acion from trained network ----------
            ac_in = (obs[np.newaxis,np.newaxis, :], jnp.array([done])[np.newaxis, :])
            assert len(ac_in[0].shape) == 3, f"{ac_in[0].shape}"
            assert len(ac_in[1].shape) == 2, f"{ac_in[1].shape}"
            hstate, pi, value = network.apply(restored_params, hstate, ac_in) 
            # hstate, pi, value = network.apply(restored_params, hstate, ac_in) # TODO does hstate need to be from the out?
            # hstate, pi, value = network.apply(runner_state.train_state.params, hstate, ac_in)
            action = pi.sample(seed=rng).round().astype(jnp.int32)[0,0,:].clip( 0, None)
            # ---------- acion from trained network ----------
            # ==================== ACTION ====================    
            print(f"-------------\nPPO {i}th actions are: {action} with sum {action.sum()}")
            start=time.time()
            obs,state,reward,done,info=env.step(key_step, state,action, env_params)
            print(f"Time for {i} step: \n",time.time()-start)
            # done, state.quant_executed
            print("{" + ", ".join([f"'{k}': {v}" for k, v in info.items()]) + "}")
            if done:
                break
        return info['window_index'],info['average_price']
    def get_twap_average_price(rngInitNum):
        rng = jax.random.PRNGKey(rngInitNum)
        rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)

        env=ExecutionEnv(ATFolder,"sell")
        env_params=env.default_params
        print(env_params.message_data.shape, env_params.book_data.shape)

        start=time.time()
        obs,state=env.reset(key_reset,env_params)
        # print("State after reset: \n",state)
        print("Time for reset: \n",time.time()-start)
        print(env_params.message_data.shape, env_params.book_data.shape)

        i = 1
        for i in range(1,10000):
            # ==================== ACTION ====================
            # ---------- acion from given strategy  ----------
            print("---"*20)
            print("window_index ",state.window_index)
            key_policy, _ = jax.random.split(key_policy,2)
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
            # if i == 277:
            #     jax.debug.breakpoint()
            def twap(state, env_params):
                # ---------- ifMarketOrder ----------
                remainingTime = env_params.episode_time - jnp.array((state.time-state.init_time)[0], dtype=jnp.int32)
                marketOrderTime = jnp.array(60, dtype=jnp.int32) # in seconds, means the last minute was left for market order
                ifMarketOrder = (remainingTime <= marketOrderTime)
                print(f"{i} remainingTime{remainingTime} marketOrderTime{marketOrderTime}")
                # ---------- ifMarketOrder ----------
                # ---------- quants ----------
                remainedQuant = state.task_to_execute - state.quant_executed
                remainedStep = state.max_steps_in_episode - state.step_counter
                stepQuant = jnp.ceil(remainedQuant/remainedStep).astype(jnp.int32) # for limit orders
                # quants = [stepQuant - 3*stepQuant//4,stepQuant//4, stepQuant//4, stepQuant//4]
                # quants = jax.random.permutation(key_policy, jnp.array([stepQuant - 3*stepQuant//4,stepQuant//4, stepQuant//4, stepQuant//4]), independent=True)
                # limit_quants_agressive = jax.random.permutation(key_policy, jnp.array([stepQuant - 3*stepQuant//4,stepQuant//4, stepQuant//4,stepQuant//4]), independent=True)
                # limit_quants_agressive = jax.random.permutation(key_policy, jnp.array([stepQuant - 3*stepQuant//4,stepQuant//4, stepQuant//4, stepQuant//4]), independent=True).at[-1].set(max(stepQuant - 3*stepQuant//4,stepQuant//4))
                limit_quants_agressive = jnp.append(jax.random.permutation(key_policy, jnp.array([stepQuant - 3*stepQuant//4,stepQuant//4, stepQuant//4]), independent=True),max(stepQuant - 3*stepQuant//4,stepQuant//4))
                # limit_quants_passive = jax.random.permutation(key_policy, jnp.array([remainedQuant - 3*remainedQuant//4,remainedQuant//4, remainedQuant//4,remainedQuant//4]), independent=True)
                limit_quants_passive = jnp.append(jax.random.permutation(key_policy, jnp.array([remainedQuant - 3*remainedQuant//4,remainedQuant//4, remainedQuant//4]), independent=True),remainedQuant//4)
                limit_quants = jnp.where(limit_quants_agressive.sum() <= remainedQuant,limit_quants_agressive,limit_quants_passive)
                # assert limit_quants.sum() <= remainedQuant, f"{limit_quants}, {limit_quants.sum()} should less_equal than {remainedQuant}"
                # if not limit_quants_agressive.sum() <= remainedQuant: print("+++")
                market_quants = jnp.array([remainedQuant - 3*remainedQuant//4,remainedQuant//4, remainedQuant//4, remainedQuant//4])
                quants = jnp.where(ifMarketOrder,market_quants,limit_quants)
                # ---------- quants ----------
                return jnp.array(quants)
                
            twap_action = twap(state, env_params)
            
            
            # print(f"Sampled {i}th actions are: ",test_action)
            print(f"Sampled {i}th actions are: ",twap_action)
            start=time.time()
            # obs,state,reward,done,info=env.step(key_step, state,test_action, env_params)
            obs,state,reward,done,info=env.step(key_step, state,twap_action, env_params)
            print(f"State after {i} step: \n",state,done,file=open('output.txt','a'))
            print(f"Time for {i} step: \n",time.time()-start)
            print("excuted ",info["quant_executed"])
            if done:
                break
        return info['window_index'], info['average_price']
    def get_advantage(rngInitNum):
        window_index1,twap=get_twap_average_price(rngInitNum)
        window_index2,ppo=get_ppo_average_price(rngInitNum)
        assert window_index1 == window_index2
        return window_index1, (ppo-twap)/twap*10000, ppo, twap
    # result_list = [get_advantage(rngInitNum) for rngInitNum in range(100)]
    for rngInitNum in range(100):
        result_tuple = get_advantage(rngInitNum) 
        print(f"window_index {result_tuple[0]} , advantage {result_tuple[1]} , ppoAP {result_tuple[2]} , twapAP {result_tuple[3]}",file=open('comparison.txt','a'))
        