import time
import jax
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from gymnax_exchange.jaxen.marl_env import MARLEnv
from gymnax_exchange.jaxob.jaxob_config import MultiAgentConfig


#MMMW class: one armed bandit approach to market making
class MMMWAgent:
    
    def __init__(self, num_actions, learning_rate, initial_volume):
        self.num_actions = num_actions  # Total number of possible actions/strategies
        self.weights = jnp.ones(num_actions) / num_actions  # Initialise weights equally, summing to 1.
        self.eta = learning_rate  # Learning rate for MW updates
        self.initial_volume = initial_volume  # Total volume to distribute across actions
        
        # Store the index of the last chosen action to update its weight later
        self.last_chosen_action_idx = -1 
    
    def compute_volumes(self):
        """Allocate total volume based on current weights."""
        return self.initial_volume * self.weights ## bit confusing naming=> not volume to trade, used to weight selection.
    
    def update_weights(self, chosen_action_reward):
        """
        Update the weight of the *previously chosen action* using its reward.
        Other weights remain unchanged for this update step.
        """
        if self.last_chosen_action_idx != -1:
            # Create a zero array for payoffs, then set the payoff for the chosen action
            payoffs_for_update = jnp.zeros(self.num_actions)
            payoffs_for_update = payoffs_for_update.at[self.last_chosen_action_idx].set(chosen_action_reward)

            new_weights = self.weights.at[self.last_chosen_action_idx].set(
                self.weights[self.last_chosen_action_idx] * jnp.exp(self.eta * chosen_action_reward)
            )
            
            # Normalize all weights so they still sum to 1
            self.weights = new_weights / jnp.sum(new_weights)
        
        # Reset last_chosen_action_idx after update
        self.last_chosen_action_idx = -1 # Or set it in act() after choosing
            
    def choose_action(self):
        """
        Choose an action based on the current weights.
        Using categorical sampling (proportional to weights).
        """
        # Ensure weights sum to 1 for jnp.random.choice
        probabilities = self.weights / jnp.sum(self.weights) 
        
        # Choose an action index based on these probabilities
        chosen_action_idx = jnp.array(jnp.random.choice(self.num_actions, p=probabilities))
        
        # Store the chosen action index for the next update step
        self.last_chosen_action_idx = chosen_action_idx 
        
        return chosen_action_idx

##initlaise the MMMW agent
mmmw_agent = MMMWAgent(num_actions=6, learning_rate=0.01, initial_volume=100)

def get_action(reward):
    
    # Update weights based on last steps reward
    if mmmw_agent.last_chosen_action_idx != -1: # Checks if the agent has a recorded action to update
        # Get the reward for the previously chosen action
        previous_action_reward = reward
        
        # Update the weights
        mmmw_agent.update_weights(previous_action_reward)
    
    #Choose next action
    chosen_action_idx = mmmw_agent.choose_action()
    
    return chosen_action_idx


#TWAP control from TWAP script
def random_policy(env, obs, rng):
    rng, _rng = jax.random.split(rng)
    return env.action_space().sample(_rng)

def twap_aggr(env, obs, rng):
    frontloading = 0.1
    steps_left = calc_steps_left(env, obs, frontloading)
    step_quant = obs['remaining_quant'] // steps_left
    action = jnp.zeros((env.n_actions,)).at[0].set(step_quant)
    return action

def twap_pass(env, obs, rng):
    frontloading = 0.1
    overstuff_factor = 2
    steps_left = calc_steps_left(env, obs, frontloading)
    step_quant = (obs['remaining_quant'] // steps_left) * overstuff_factor
    action = jnp.zeros((env.n_actions,)).at[-1].set(step_quant)
    return action

def all_passive(env, obs, rng):
    return jnp.zeros((env.n_actions,)).at[-1].set(obs['remaining_quant'])

def calc_steps_left(env, obs, frontloading):
    if env.ep_type == "fixed_steps":
        steps_left = obs["max_steps"] - obs["step_counter"]
    elif env.ep_type == "fixed_time":
        steps_left = jax.lax.cond(
            obs['delta_time'] == 0,
            lambda: obs["max_steps"] - obs["step_counter"],
            lambda: (obs["time_remaining"] // obs['delta_time']).astype(jnp.int32),
        )
    steps_left = jnp.clip((steps_left * (1-frontloading)).astype(jnp.int32), 1, None)
    return steps_left




if __name__ == "__main__":

    multi_agent_config = MultiAgentConfig()

    rng = jax.random.PRNGKey(30) # TODO i think this should be changed to the new key function in JAX .key()
    rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)

    # Instantiate the MARL environment.
    env = MARLEnv(
        key=key_reset,
        multi_agent_config=multi_agent_config,
    )

    # Get the default combined parameters.
    print("starting default parameters")
    env_params = env.default_params

    # Reset the environment.
    obs, state = env.reset(key_reset, env_params)
    

    #print("State after reset: \n",state)
    print("Inventory after reset: \n",state.inventory)

    

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

        if done:
            print("==="*20)
