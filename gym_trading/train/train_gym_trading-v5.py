# %% ==========================================================================
import time
import gym
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure

from gym_trading import utils
from gym_trading.envs.broker import Flag
from gym_trading.envs.base_environment import BaseEnv
from gym_trading.data.data_pipeline import ExternalData
from gym_trading.data.data_pipeline import Debug; 
Debug.if_return_single_flie = False # if True then return Flow_list # Flase if you want to debug
Debug.if_return_part_data = True # default for debugging
Debug.if_whole_data = False # load the whole dataset
import warnings; warnings.filterwarnings("ignore") # clear warnings
string = time.ctime().replace(" ","-").replace(":","-"); Flag.log(log_string = "/Users/kang/GitHub/NeuralLOB/venv_rnn-v5/Oct_05/rnn_ppo_gym_trading-"+string)
# check_env(env)
# num_cpu = 10; venv = DummyVecEnv([lambda: Monitor(gym.make("GymTrading-v1", Flow = Flow))] * num_cpu)


Flow = ExternalData.get_sample_order_book_data()
monitord_env = Monitor(
    env = gym.make("OptimalLiquidation-v2",Flow = Flow),
    # env = gym.make("GymTrading-v1",Flow = Flow),
              )
venv = DummyVecEnv([lambda: monitord_env])

# # %% training strategies 1

# model = RecurrentPPO(
#     "MlpLstmPolicy", 
#     venv, 
#     verbose=1,
#     learning_rate = utils.biquadrate_schedule(3e-4),
#     tensorboard_log="/Users/kang/GitHub/NeuralLOB/venv_rnn-v5/Oct_05/")
# # initial model


# model.learn(total_timesteps=int(1e5), tb_log_name="RNN_PPO_initial",callback=utils.TensorboardCallback())
# for i in range(int(3e3)):
#     model.learn(total_timesteps=int(1e6), tb_log_name="RNN_PPO_improve",reset_num_timesteps=False,callback=utils.TensorboardCallback())
#     string = time.ctime().replace(" ","-").replace(":","-")
#     model.save("/Users/kang/GitHub/NeuralLOB/venv_rnn-v5/Oct_05/rnn_ppo_gym_trading-"+string)
#     Flag.log(log_string = "/Users/kang/GitHub/NeuralLOB/venv_rnn-v5/Oct_05/rnn_ppo_gym_trading-"+string)
# # model learning and logging

# %% training strategies 2

model = RecurrentPPO(
    "MlpLstmPolicy", 
    venv, 
    verbose=1,
    learning_rate = utils.linear_schedule(1e-3),
    tensorboard_log="/Users/kang/GitHub/NeuralLOB/venv_rnn-v5/Oct_05/")
# initial model


model.learn(
    total_timesteps=int(1e8), 
    tb_log_name="RNN_PPO_initial",
    eval_env = venv,
    callback=utils.TensorboardCallback()
    )
# for i in range(int(3e3)):
#     model.learn(total_timesteps=int(1e6), tb_log_name="RNN_PPO_improve",reset_num_timesteps=False,callback=utils.TensorboardCallback())
#     string = time.ctime().replace(" ","-").replace(":","-")
#     model.save("/Users/kang/GitHub/NeuralLOB/venv_rnn-v5/Oct_05/rnn_ppo_gym_trading-"+string)
#     Flag.log(log_string = "/Users/kang/GitHub/NeuralLOB/venv_rnn-v5/Oct_05/rnn_ppo_gym_trading-"+string)
# # model learning and logging

# %% test the train result
from stable_baselines3.common.evaluation import evaluate_policy
eval_env = gym.make("GymTrading-v1",Flow = Flow) ## TODO
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
# model testing(loaded from the trained)

model = RecurrentPPO.load("/Users/kang/GitHub/NeuralLOB/tensorboard_rnn/rnn_ppo_gym_trading-v1Wed-Aug-31-19-58-55-2022.zip")
start = time.time()
env = gym.make("GymTrading-v1",Flow = Flow) ## TODO
# model testing(loaded from the saved)
















