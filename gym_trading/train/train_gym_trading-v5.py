# %% ==========================================================================
import time
import gym
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from gym_trading import utils
from gym_trading.envs.broker import Flag
from gym_trading.envs.base_environment import BaseEnv
from gym_trading.data.data_pipeline import ExternalData
from gym_trading.data.data_pipeline import Debug; 
Debug.if_return_single_flie = False # if True then return Flow_list # Flase if you want to debug
Debug.if_whole_data = False # load the whole dataset
import warnings; warnings.filterwarnings("ignore") # clear warnings


Flow = ExternalData.get_sample_order_book_data()
env = gym.make("GymTrading-v1",Flow = Flow)
# check_env(env)

num_cpu = 10; venv = DummyVecEnv([lambda: Monitor(gym.make("GymTrading-v1", Flow = Flow))] * num_cpu)
# %%
string = time.ctime().replace(" ","-").replace(":","-")
Flag.log(log_string = "/Users/kang/GitHub/NeuralLOB/venv_rnn-v5/Sep_26/rnn_ppo_gym_trading-"+string)

model = RecurrentPPO(
    "MlpLstmPolicy", 
    env, 
    verbose=1,
    learning_rate = utils.biquadrate_schedule(3e-4),
    tensorboard_log="/Users/kang/GitHub/NeuralLOB/venv_rnn-v5/Sep_26/")

model.learn(total_timesteps=int(1e6), tb_log_name="RNN_PPO_initial")
for i in range(int(3e3)):
    model.learn(total_timesteps=int(1e6), tb_log_name="RNN_PPO_improve",reset_num_timesteps=False)
    string = time.ctime().replace(" ","-").replace(":","-")
    model.save("/Users/kang/GitHub/NeuralLOB/venv_rnn-v5/Sep_26/rnn_ppo_gym_trading-"+string)
    Flag.log(log_string = "/Users/kang/GitHub/NeuralLOB/venv_rnn-v5/Sep_26/rnn_ppo_gym_trading-"+string)

# %% test the train result
model = RecurrentPPO.load("/Users/kang/GitHub/NeuralLOB/tensorboard_rnn/rnn_ppo_gym_trading-v1Wed-Aug-31-19-58-55-2022.zip")
start = time.time()
env = gym.make("GymTrading-v1",Flow = Flow) ## TODO

















