# from jax import config
# config.update("jax_enable_x64",True)
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import time
from flax.linen.initializers import constant, orthogonal
from typing import Optional, Sequence, NamedTuple, Any, Dict
from flax.training.train_state import TrainState
import distrax
import gymnax
import functools
from gymnax.environments import spaces
import sys
import chex
sys.path.append('../purejaxrl')
sys.path.append('../AlphaTrade')
from purejaxrl.wrappers import FlattenObservationWrapper, LogWrapper,ClipAction, VecEnv,NormalizeVecObservation,NormalizeVecReward
from gymnax_exchange.jaxen.exec_env import ExecutionEnv
import os
import flax
from jax.lib import xla_bridge 
print(xla_bridge.get_backend().platform)
#Code snippet to disable all jitting.
from jax import config
config.update("jax_disable_jit", False) 
# config.update("jax_disable_jit", True)
config.update("jax_check_tracer_leaks", False) #finds a whole assortment of leaks if true... bizarre.
import datetime
jax.numpy.set_printoptions(linewidth=250)



wandbOn = True # False
if wandbOn:
    import wandb

def save_checkpoint(params, filename):
    with open(filename, 'wb') as f:
        f.write(flax.serialization.to_bytes(params))
        print(f"Checkpoint saved to {filename}")


class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell()(rnn_state, ins)
        # try: add another rnn layer
        new_rnn_state, y = nn.GRUCell()(new_rnn_state, y)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        return nn.GRUCell.initialize_carry(
            jax.random.PRNGKey(0), (batch_size,), hidden_size
        )

class MultiVariateNormalDiagClipped(distrax.MultivariateNormalDiag):
    def __init__(
            self,
            loc: Optional[jax.Array] = None,
            scale_diag: Optional[jax.Array] = None,
            max_scale_diag: Optional[jax.Array] = None,
        ):
        super().__init__(loc, scale_diag)
        self.max_scale_diag = max_scale_diag

    def __getitem__(self, index) -> distrax.MultivariateNormalDiag:
        """See `Distribution.__getitem__`."""
        index = distrax.distribution.to_batch_shape_index(self.batch_shape, index)
        return MultiVariateNormalDiagClipped(
            loc=self.loc[index],
            scale_diag=jnp.min(self.max_scale_diag, self.scale_diag[index]),
            max_scale_diag=self.max_scale_diag,
        )


class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        embedding = nn.Dense(
            128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.LayerNorm()(embedding)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)
        embedding = nn.LayerNorm()(embedding)

        actor_net = nn.Dense(128, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        actor_net = nn.LayerNorm()(actor_net)
        actor_net = nn.relu(actor_net)
        
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
            # self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.5)
        )(actor_net)
        max_action_logstd = -1.6  # exp -1.6 = 0.2
        actor_logtstd = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(max_action_logstd), name="log_std"
        )(actor_net)
        # actor_logtstd = self.param("log_std", nn.initializers.constant(-1.6), (self.action_dim,))
        #Trying to get an initial std_dev of 0.2 (log(0.2)~=-0.7)
        # pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))
        pi = MultiVariateNormalDiagClipped(
            actor_mean * self.config['MAX_TASK_SIZE'],
            jnp.exp(actor_logtstd) * self.config['MAX_TASK_SIZE'],
            jnp.exp(actor_logtstd) * self.config['MAX_TASK_SIZE']
        )

        critic = nn.Dense(128, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.LayerNorm()(critic)
        critic = nn.relu(critic)

        critic = nn.Dense(64, kernel_init=orthogonal(1), bias_init=constant(0.0))(
            critic
        )
        critic = nn.LayerNorm()(critic)
        critic = nn.relu(critic)

        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    

    env = ExecutionEnv(
        config["ATFOLDER"],
        config["TASKSIDE"],
        config["WINDOW_INDEX"],
        config["ACTION_TYPE"],
        config["TASK_SIZE"],
        config["REWARD_LAMBDA"],
        config["DATA_TYPE"],
    )
    env_params = env.default_params
    env = LogWrapper(env)    
    
    if config["NORMALIZE_ENV"]:
        env = NormalizeVecObservation(env)
        # NOTE: don't normalize reward for now
        # env = NormalizeVecReward(env, config["GAMMA"])
    

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        network = ActorCriticRNN(env.action_space(env_params).shape[0], config=config)
        rng, _rng = jax.random.split(rng)
        init_x = (
            jnp.zeros(
                (1, config["NUM_ENVS"], *env.observation_space(env_params).shape)
            ),
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], 128)
        network_params = network.init(_rng, init_hstate, init_x)
        
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule,b1=0.9,b2=0.99, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"],b1=0.9,b2=0.99, eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )
        
        # jax.debug.breakpoint()
        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], 128)

        # TRAIN LOOP
        def _update_step(runner_state, unused):

            """
            Pseudocode
            if i%50 ==0:
                envparam.message_data.reshuffled()
                envparam.book_data.reshuffled()
            
            reshuffled():
                0-30,30-60.... --> 5-35,35-65
                                    ...or 40-70,70-100
            
            """

            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, last_done, hstate, rng = runner_state
                rng, _rng = jax.random.split(rng)

                # SELECT ACTION
                ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
                hstate, pi, value = network.apply(train_state.params, hstate, ac_in)


                action = pi.sample(seed=_rng) # 4*1, should be (4*4: 4actions * 4envs)
                # Guess to be 4 actions. caused by ppo_rnn is continuous. But our action space is discrete
                log_prob = pi.log_prob(action)

                value, action, log_prob = (
                    value.squeeze(0),
                    action.squeeze(0),
                    log_prob.squeeze(0),
                )

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                obsv_step, env_state_step, reward_step, done_step, info_step = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action, env_params)
                transition = Transition(
                    done_step, action, value, reward_step, log_prob, last_obs, info_step
                )
                runner_state = (train_state, env_state_step, obsv_step, done_step, hstate, rng)
                return runner_state, transition

            initial_hstate = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_done, hstate, rng = runner_state
            ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
            _, _, last_val = network.apply(train_state.params, hstate, ac_in)
            last_val = last_val.squeeze(0)
            last_val = jnp.where(last_done, jnp.zeros_like(last_val), last_val)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        # RERUN NETWORK
                        _, pi, value = network.apply(
                            params, init_hstate[0], (traj_batch.obs, traj_batch.done)
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, init_hstate, traj_batch, advantages, targets
                    )
                    # jax.debug.print("grads: {}", grads['params']['log_std'])
                    # jax.debug.print("grads: {}", grads)
                    # print(jax.tree_util.tree_structure(grads))
                    train_state = train_state.apply_gradients(grads=grads)
                    # make sure the action std doesn't grow too large
                    # train_state.params = train_state.params.copy({
                    #     'log_std': jnp.min(
                    #         train_state.params["log_std"],
                    #         -1.6 * jnp.ones_like(train_state.params["log_std"]),
                    #     )
                    # })
                    return train_state, total_loss

                (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state

                rng, _rng = jax.random.split(rng)
                permutation = jax.random.permutation(_rng, config["NUM_ENVS"])
                batch = (init_hstate, traj_batch, advantages, targets)

                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1]
                            + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

            init_hstate = initial_hstate[None, :]  # TBH
            update_state = (
                train_state,
                init_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = (traj_batch.info,train_state.params)
            rng = update_state[-1]
            if config.get("DEBUG"):

                def callback(metric):
                    
                    info,trainstate_params=metric
                    
                    return_values = info["returned_episode_returns"][
                        info["returned_episode"]
                    ]
                    timesteps = (
                        info["timestep"][info["returned_episode"]] * config["NUM_ENVS"]
                    )
                    
                    def evaluation():
                        if not os.path.exists(config['CHECKPOINT_DIR']): os.makedirs(config['CHECKPOINT_DIR'])
                        # Inside your loop or function where you save the checkpoint
                        if any(timesteps % int(1e3) == 0) and len(timesteps) > 0:  # +1 since global_step is 0-indexed
                            start = time.time()
                            jax.debug.print(">>> checkpoint saving {}",round(timesteps[0], -3))
                            # Save the checkpoint to the specific directory
                            checkpoint_filename = os.path.join(config['CHECKPOINT_DIR'], f"checkpoint_{round(timesteps[0], -3)}.ckpt")
                            save_checkpoint(trainstate_params, checkpoint_filename)  # Assuming trainstate_params contains your model's state
                            jax.debug.print("+++ checkpoint saved  {}",round(timesteps[0], -3))
                            jax.debug.print("+++ time taken        {}",time.time()-start)        
                    evaluation()
                    
                    revenues = info["total_revenue"][info["returned_episode"]]
                    quant_executed = info["quant_executed"][info["returned_episode"]]
                    average_price = info["average_price"][info["returned_episode"]]
                    
                    # slippage_rm = info["slippage_rm"][info["returned_episode"]]
                    # price_drift_rm = info["price_drift_rm"][info["returned_episode"]]
                    # price_adv_rm = info["price_adv_rm"][info["returned_episode"]]
                    # vwap_rm = info["vwap_rm"][info["returned_episode"]]
                    
                    current_step = info["current_step"][info["returned_episode"]]
                    # advantage_reward = info["advantage_reward"][info["returned_episode"]]
                    
                    '''
                    print(info["current_step"][0,0],info["total_revenue"][0,0],info["average_price"][0,0],info['quant_executed'][0,0],info['action'][0,0])  
                    if info['done']: print("==="*10 + str(info["window_index"]) +"==="*10 + '\n')      
                    # if info['done']: print("==="*10 + "==="*10 + '\n')      
                    # if info['done']: print("==="*10 + str(info["window_index"])[0,0] + "==="*10 + '\n')      
                    # print(info["total_revenue"])  
                    # print(info["quant_executed"])   
                    # print(info["average_price"])   
                    # print(info["returned_episode_returns"])
                    '''
                    
                    # '''
                    # NOTE: only log every 100th timestep
                    for t in range(0, len(timesteps), 100):
                    # for t in range(len(timesteps)):
                        if wandbOn:
                            wandb.log(
                                data={
                                    "global_step": timesteps[t],
                                    "episodic_return": return_values[t],
                                    "episodic_revenue": revenues[t],
                                    "quant_executed":quant_executed[t],
                                    "average_price":average_price[t],
                                    # "slippage_rm":slippage_rm[t],
                                    # "price_adv_rm":price_adv_rm[t],
                                    # "price_drift_rm":price_drift_rm[t],
                                    # "vwap_rm":vwap_rm[t],
                                    "current_step":current_step[t],
                                    # "advantage_reward":advantage_reward[t],
                                },
                                commit=True
                            )        
                        else:
                            print(
                                # f"global step={timesteps[t]:<11} | episodic return={return_values[t]:.10f<15} | episodic revenue={revenues[t]:.10f<15} | average_price={average_price[t]:<15}"
                                f"global step={timesteps[t]:<11} | episodic return={return_values[t]:<20} | episodic revenue={revenues[t]:<20} | average_price={average_price[t]:<11}"
                            )     
                            # print("==="*20)      
                            # print(info["current_step"])  
                            # print(info["total_revenue"])  
                            # print(info["quant_executed"])   
                            # print(info["average_price"])   
                            # print(info["returned_episode_returns"])
                    # '''


                jax.debug.callback(callback, metric)

            runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            jnp.zeros((config["NUM_ENVS"]), dtype=bool),
            init_hstate,
            _rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metric": metric}

    return train

if __name__ == "__main__":
    timestamp=datetime.datetime.now().strftime("%m-%d_%H-%M")

    ppo_config = {
        "LR": 5e-4, # 5e-4, #5e-5, #1e-4,#2.5e-5,
        "ENT_COEF": 0.0, #0.1, 0.01
        "NUM_ENVS": 1024, #1024, #128, #64, 1000,
        "TOTAL_TIMESTEPS": 1e8,  #5e7, # 50MIL for single data window convergence #,1e8,  # 6.9h
        "NUM_MINIBATCHES": 8, #8, 4, 2,
        "UPDATE_EPOCHS": 30, #5,
        "NUM_STEPS": 1024, #500,
        "CLIP_EPS": 0.2,
        
        "GAMMA": 0.99,
        "GAE_LAMBDA": 1.0, #0.95,
        "VF_COEF": 1.0, #0.5,
        "MAX_GRAD_NORM": 0.5,# 2.0,
        "ANNEAL_LR": True, #True,
        "NORMALIZE_ENV": True,  # only norms observations (not reward)
        
        "ACTOR_TYPE": "RNN",
        "MAX_TASK_SIZE": 500,
        
        "ENV_NAME": "alphatradeExec-v0",
        "WINDOW_INDEX": 2, # 2 fix random episode #-1,
        "DEBUG": True,
        
        "TASKSIDE": "sell",
        "REWARD_LAMBDA": 1., #0.001,
        "ACTION_TYPE": "pure", # "delta"
        "TASK_SIZE": 500, # 500,
        "DATA_TYPE": "fixed_time", # "fixed_time", "fixed_steps"
      
        "ATFOLDER": "./training_oneDay", #"/homes/80/kang/AlphaTrade/training_oneDay/",
        # "ATFOLDER": "./training_oneMonth", #"/homes/80/kang/AlphaTrade/training_oneDay/",
        "RESULTS_FILE": "training_runs/results_file_"+f"{timestamp}",  # "/homes/80/kang/AlphaTrade/results_file_"+f"{timestamp}",
        "CHECKPOINT_DIR": "training_runs/checkpoints_"+f"{timestamp}",  # "/homes/80/kang/AlphaTrade/checkpoints_"+f"{timestamp}",
    }

    if wandbOn:
        run = wandb.init(
            project="AlphaTradeJAX_Train",
            config=ppo_config,
            # sync_tensorboard=True,  # auto-upload  tensorboard metrics
            save_code=True,  # optional
        )
        import datetime;params_file_name = f'params_file_{wandb.run.name}_{timestamp}'
    else:
        import datetime;params_file_name = f'params_file_{timestamp}'

    print(f"Results will be saved to {params_file_name}")
    
    # +++++ Single GPU +++++
    rng = jax.random.PRNGKey(0)
    # rng = jax.random.PRNGKey(30)
    train_jit = jax.jit(make_train(ppo_config))
    start=time.time()
    out = train_jit(rng)
    print("Time: ", time.time() - start)
    # +++++ Single GPU +++++

    # # +++++ Multiple GPUs +++++
    # num_devices = 4
    # rng = jax.random.PRNGKey(30)
    # rngs = jax.random.split(rng, num_devices)
    # train_fn = lambda rng: make_train(ppo_config)(rng)
    # start=time.time()
    # out = jax.pmap(train_fn)(rngs)
    # print("Time: ", time.time()-start)
    # # +++++ Multiple GPUs +++++
    
    

    # '''
    # # ---------- Save Output ----------
    import flax

    train_state = out['runner_state'][0] # runner_state.train_state
    params = train_state.params
    


    import datetime;params_file_name = f'params_file_{wandb.run.name}_{datetime.datetime.now().strftime("%m-%d_%H-%M")}'

    # Save the params to a file using flax.serialization.to_bytes
    with open(params_file_name, 'wb') as f:
        f.write(flax.serialization.to_bytes(params))
        print(f"params saved")

    # Load the params from the file using flax.serialization.from_bytes
    with open(params_file_name, 'rb') as f:
        restored_params = flax.serialization.from_bytes(flax.core.frozen_dict.FrozenDict, f.read())
        print(f"params restored")
        
    # jax.debug.breakpoint()
    # assert jax.tree_util.tree_all(jax.tree_map(lambda x, y: (x == y).all(), params, restored_params))
    # print(">>>")
    # '''

    if wandbOn:
        run.finish()
