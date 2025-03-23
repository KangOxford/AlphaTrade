import os
import sys
sys.path.append(os.path.abspath('/home/duser/AlphaTrade'))
sys.path.append('.')
import time
import dataclasses
import jax
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from gymnax_exchange.jaxen.mm_env import MarketMakingEnv
from gymnax_exchange.jaxob.jaxob_config import EnvironmentConfig
import faulthandler
import pandas as pd  
import chex

faulthandler.enable()

class MMMWAgent:
    def __init__(self, num_actions, learning_rate, initial_volume):
        self.num_actions = num_actions
        self.weights = jnp.ones(num_actions)*10 // num_actions  # Initialize equal weights
        self.eta = learning_rate  # Learning rate for MW updates
        self.initial_volume = initial_volume  # Total volume to distribute across actions
    
    def compute_volumes(self):
        """Allocate volumes based on weights."""
        return self.initial_volume * self.weights
    
    def update_weights(self, payoffs):
        """Update weights using the Multiplicative Weights Update rule."""
        new_weights = self.weights * jnp.exp(self.eta * payoffs)
        self.weights = new_weights / np.sum(new_weights)  # Normalize to sum to 1
        #jax.debug.print("self.weights :{}",self.weights)
    def act(self, prices,quants, mid_price):
        """
        Decide action quantities based on weights and previous trade outcomes.
        
        Arguments:
        - prices: Array of 6 action prices (corresponding to bid/ask levels).
        - trades: Array of trade data (n_messages x 8).
        - mid_prices: Average mid-prices for the current step.
        
        Returns:
        - action_volumes: Quantities to offer at each action price.
        """
        # Compute payoffs for each action
        payoffs = np.zeros(self.num_actions)
        for i in range(self.num_actions):
         payoff = jnp.abs(prices[i] - mid_price) * jnp.abs(quants[i])
         payoffs[i] = payoff
       # jax.debug.print("payoffs:{}",payoffs)
        payoffs_norm=payoffs/(jnp.sum(payoffs)+0.0001)# normalise, small add to stabilise
  
      #  jax.debug.print("payoffs_norm:{}",payoffs_norm)
         
        
        # Update weights using the payoffs
        self.update_weights(payoffs_norm)

        # Compute action volumes based on weights
        action_volumes = self.compute_volumes()
        weights=self.weights
       # jax.debug.print("weights:{}",weights)
        return action_volumes,weights
# Initialize the agent
mmmw_agent = MMMWAgent(num_actions=6, learning_rate=0.01, initial_volume=100)

# During each step
def get_action(env_state):
    # Compute action prices from the state
    action_prices=env_state.prev_executed[:,0]
    action_quants=env_state.prev_executed[:,1]

    mid_prices = env_state.mid_price  

    # Agent decides action volumes
    action_volumes,weights = mmmw_agent.act(action_prices, action_quants, mid_prices)
    action_volumes=jnp.array(action_volumes).astype(jnp.int32)
    #jax.debug.print("weights :{}",weights)
    return action_volumes, weights

wandbOn = True
if wandbOn:
    import wandb
''''
This agent needs fixed prices action space. Need to debug to check it runs, but I have no gpu... :(
'''

if __name__ == "__main__":
    try:
        ATFolder = sys.argv[1]
        print("AlphaTrade folder:",ATFolder)
    except:
        # ATFolder = "./testing_oneDay"
        #ATFolder = "/training_oneDay"
        #ATFolder = "/home/duser/AlphaTrade/training_oneDay/train"
        ATFolder= "/home/duser/AlphaTrade/testing"

        # ATFolder = '/home/duser/AlphaTrade'
        # ATFolder = '/homes/80/kang/AlphaTrade'
        # ATFolder = "/homes/80/kang/AlphaTrade/testing_oneDay"
        # ATFolder = "/homes/80/kang/AlphaTrade/training_oneDay"
        # ATFolder = "/homes/80/kang/AlphaTrade/testing"
    config = {
        "ATFOLDER": ATFolder,
        "WINDOW_INDEX": 0,
        "REWARD_LAMBDA": 0.1,
        "EP_TYPE": "fixed_time",
        "EPISODE_TIME": 60*30,  
    }
        
    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)
    
    # env=MarketMakingEnv(ATFolder,"sell",1)

    env_cfg = EnvironmentConfig()

    env = MarketMakingEnv(
        cfg = env_cfg,
        key = key_reset,
        alphatradePath=config["ATFOLDER"],
        window_index=config["WINDOW_INDEX"],
        episode_time=config["EPISODE_TIME"],
        ep_type=config["EP_TYPE"],
    )
    # env_params=env.default_params
    env_params = dataclasses.replace(
        env.default_params,
        reward_lambda=0.00001,
        episode_time=config["EPISODE_TIME"],  # in seconds
    )
    # print(env_params.message_data.shape, env_params.book_data.shape)


    start=time.time()
    obs,state=env.reset(key_reset, env_params)
    print("Time for reset: \n",time.time()-start)

    #print("State after reset: \n",state)
    print("Inventory after reset: \n",state.inventory)

    if wandbOn:
        run = wandb.init(
            project="MarketMaking_MMMW_test",
            config=config,
            # sync_tensorboard=True,  # auto-upload  tensorboard metrics
            save_code=False,  # optional
        )
    

    # print(env_params.message_data.shape, env_params.book_data.shape)
    for i in range(1,2000):
         # ==================== ACTION ====================
        # ---------- acion from random sampling ----------
        print("-"*200)
        key_policy, _ = jax.random.split(key_policy, 2)
        key_step, _ = jax.random.split(key_step, 2)
        action, weights=get_action(state)
        start=time.time()
        obs, state, reward, done, info = env.step(
            key_step, state, action, env_params)
        if config.get("DEBUG"):
                def callback(info,weights):
                    #------------Collect info for plotting---------------------------#

                    #-----------Info----------#
                    for i in weights.shape[0]:
                        wandb.log(
                            data={                          
                                f"action_{int(i)}":weights[i],
                            },
                            commit=True
                        )
                    PnL = info["total_PnL"]
                    inventories = info["inventory"] 
                    buyQuant=info["buyQuant"]
                    sellQuant=info["sellQuant"]
                    reward=info["reward"]
                    other_exec_quants=info["other_exec_quants"]
                    netWorth = info["netWorth"]
                    averageMidprice=info["averageMidprice"]
                    averageBestbid=info["average_best_bid"]
                    averageBestask=info["average_best_ask"]

                    #-----------Append to list for plotting---------------------------#
                    if wandbOn:
                        wandb.log(
                            data={                          
                                #---------Reward and error bars--------#
                                "reward":jnp.mean(reward) if reward.size > 0 else 0,
                                #---------PnL and errors bars-----------#
                                "PnL_mean": jnp.mean(PnL) if PnL.size > 0 else 0,
                                "PnL_plus_std": (jnp.mean(PnL) + jnp.std(PnL)) if PnL.size > 0 else 0,
                                "PnL_minus_std": (jnp.mean(PnL) - jnp.std(PnL)) if PnL.size > 0 else 0,
                                #-------------NetWorth and error bars----------#
                                "netWorth": jnp.mean(netWorth) if netWorth.size > 0 else 0,
                                "netWorth_plus_std": (jnp.mean(netWorth) + jnp.std(netWorth)) if netWorth.size > 0 else 0,
                                "netWorth_minus_std": (jnp.mean(netWorth) - jnp.std(netWorth)) if netWorth.size > 0 else 0,
                                
                                #----------Iventory and error bars------------#
                                "inventory_train": jnp.mean(inventories) if inventories.size > 0 else 0, 
                                "inventory_train_plus_std":(jnp.mean(inventories) + jnp.std(inventories)) if inventories.size > 0 else 0,
                                "inventory_train_minus_std":(jnp.mean(inventories) - jnp.std(inventories)) if inventories.size > 0 else 0,

                                #----------Buy and Sell Quant and error bars------------#

                                "buyQuant_train":jnp.mean(buyQuant) if buyQuant.size > 0 else 0,
                                "sellQuant_train":jnp.mean(sellQuant) if sellQuant.size > 0 else 0,
                                "other_exec_quants_train":jnp.mean(other_exec_quants) if other_exec_quants.size > 0 else 0,
                                "averageMidprice_train":jnp.mean(averageMidprice) if averageMidprice.size>0 else 0,
                                "averageBestbid_train":jnp.mean(averageBestbid) if averageBestbid.size>0 else 0,
                                "averageBestask_train":jnp.mean(averageBestask) if averageBestask.size>0 else 0,
                               
                            },
                            commit=True
                        )
                        
                        # Additionally log histograms for full distributions

                        if PnL.size > 0:
                            wandb.log({"PnL_histogram": wandb.Histogram(PnL)}, commit=False)
                        # Add networth histogram
                        if netWorth.size > 0:
                            wandb.log({"networth_histogram": wandb.Histogram(netWorth)}, commit=False)

                jax.debug.callback(callback,info)

        if done:
            print("==="*20)
