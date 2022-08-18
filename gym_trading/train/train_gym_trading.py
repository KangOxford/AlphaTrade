# %% ==========================================================================
# clear warnings
from email import policy
import warnings
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from gym_trading.envs.base_environment import BaseEnv
from gym_trading.data.data_pipeline import ExternalData



warnings.filterwarnings("ignore")
Flow = ExternalData.get_sample_order_book_data()

env = gym.make("GymTrading-v1",Flow = Flow) ## TODO
# env = gym.DummyVecEnv([lambda: gym.make("GymTrading-v1",Flow = Flow)])


{

- pretrain
    - define the fixed delta policy
    - pretrain
    - combine the policy

}


check_env(env)
model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="./ppo_gymtrading_tensorboard8/")



# %time model.learn(total_timesteps=int(1e7), n_eval_episodes = int(1e5))
# model.learn(total_timesteps=int(3e6), n_eval_episodes = int(1e5))
model.learn(total_timesteps=int(2e6), tb_log_name="first_run")
model.learn(total_timesteps=int(2e6), tb_log_name="second_run", reset_num_timesteps=False)
model.learn(total_timesteps=int(2e6), tb_log_name="third_run", reset_num_timesteps=False)
model.learn(total_timesteps=int(2e6), tb_log_name="fourth_run", reset_num_timesteps=False)
model.learn(total_timesteps=int(2e6), tb_log_name="fifth_run", reset_num_timesteps=False)
model.learn(total_timesteps=int(2e6), tb_log_name="sixth_run", reset_num_timesteps=False)
model.learn(total_timesteps=int(2e6), tb_log_name="seventh_run", reset_num_timesteps=False)
model.learn(total_timesteps=int(2e6), tb_log_name="eighth_run", reset_num_timesteps=False)
model.learn(total_timesteps=int(2e6), tb_log_name="ninth_run", reset_num_timesteps=False)



model.save("gym_trading-v1") 










