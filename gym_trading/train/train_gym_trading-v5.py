# %% ==========================================================================
import time
import gym
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

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

def linear_schedule(initial_value):
    """
    Linear learning rate schedule.
    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress):
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress: (float)
        :return: (float)
        """
        return progress * initial_value

    return func

def biquadrate_schedule(initial_value):
    if isinstance(initial_value, str):initial_value = float(initial_value)
    def func(progress):return progress * progress * progress * progress * initial_value
    return func

model = RecurrentPPO(
    "MlpLstmPolicy", 
    # venv, 
    env, 
    # n_steps = Flag.max_episode_steps,
    verbose=1,
    learning_rate = biquadrate_schedule(3e-4),
    tensorboard_log="/Users/kang/GitHub/NeuralLOB/venv_rnn-v5/Sep_26/")

model.learn(total_timesteps=int(1e6), tb_log_name="RNN_PPO_initial")
for i in range(3e3):
    model.learn(total_timesteps=int(1e6), tb_log_name="RNN_PPO_improve",reset_num_timesteps=False)
    Flag.log()
    string = time.ctime().replace(" ","-").replace(":","-")
    model.save("/Users/kang/GitHub/NeuralLOB/tensorboard_rnn-v5/Sep_26/rnn_ppo_gym_trading-"+string)

# model = RecurrentPPO(
#     "MlpLstmPolicy", 
#     venv, 
#     verbose=1,
#     # n_steps = Flag.max_episode_steps, #n_steps: The number of steps to run for each environment per update
#     learning_rate = linear_schedule(3e-4),
#     tensorboard_log="/Users/kang/GitHub/NeuralLOB/venv_rnn/")
# model.learn(total_timesteps=int(3e6), tb_log_name="RNN_PPO_init")
# model.save("/Users/kang/GitHub/NeuralLOB/tensorboard_rnn/rnn_ppo_gym_trading-v1")

# tensorboard --logdir /Users/kang/GitHub/NeuralLOB/venv_rnn-v5/









# =============================================================================

# =============================================================================





# # # %% test the train result
model = RecurrentPPO.load("/Users/kang/GitHub/NeuralLOB/tensorboard_rnn/rnn_ppo_gym_trading-v1Wed-Aug-31-19-58-55-2022.zip")
start = time.time()
env = gym.make("GymTrading-v1",Flow = Flow) ## TODO


# reward_list = []
# obs = env.reset()
# done = False
# for i in range(int(1e8)):
#     print(i)
#     action, _states = model.predict(obs)
#     obs, reward, done, info = env.step(action)
#     rewar_list.append(reward)
#     if done:
#         obs = env.reset()
#         break 


# for i in range(int(1e3)):
#     obs = env.reset()
#     # ;print(obs)
#     done = False
#     running_reward = 0
#     for i in range(int(1e8)):
#         if i//int(1e5) == i/int(1e5):
#             print("Epoch {}, testing time {}".format(Epoch,(time.time()-start)/60))
#         action, _states = model.predict(obs)
#         obs, reward, done, info = env.step(action)
#         # env.render()
#         rewar_list.append(reward)
#         if done:
#             info_list.append(info)
#             obs = env.reset()
#         rewar_list.append(reward)
#             Epoch += 1
#             break 
        
# import pandas as pd
# df = pd.DataFrame(info_list)
# string = time.ctime().replace(" ","-").replace(":","-")
# df.to_csv("info_df"+string+".csv")


# #  analyse the result
# df1=df[(df.Advantage <= 100) & (df.Advantage>=-100)]
# grouped = df1.groupby(df.Left)    
# df2=grouped.mean()
# df3 = df2.set_index("Left").drop(["Step"],axis =1)
# df3.plot()
# fig = df3.plot().get_figure()
# fig.savefig(string+'.png', dpi=300)

# # %% test the twap result
# start = time.time()
# env = gym.make("GymTrading-v1",Flow = Flow) ## TODO

# info_list = []
# Epoch = 0
# for i in range(int(1e3)):
#     obs = env.reset()
#     # ;print(obs)
#     done = False
#     running_reward = 0
#     for i in range(int(1e8)):
#         if i//int(1e5) == i/int(1e5):
#             print("Epoch {}, testing time {}".format(Epoch,(time.time()-start)/60))
#         action = Flag.num2liquidate//Flag.max_episode_steps + 1
#         obs, reward, done, info = env.step(action)
#         # env.render()
#         if done:
#             info_list.append(info)
#             obs = env.reset()
#             Epoch += 1
#             break 
# import pandas as pd
# df = pd.DataFrame(info_list)
# string = time.ctime().replace(" ","-").replace(":","-")
# df.to_csv("info_df"+string+".csv")



# # %% difference between to algorithms
# #  analyse the result
# def analyse_df(df):
#     df1=df[(df.Advantage <= 100) & (df.Advantage>=-100)]
#     grouped = df1.groupby(df.Left)    
#     df2=grouped.mean()
#     df3 = df2.set_index("Left").drop(["Step"],axis =1)
#     df3.drop(["Unnamed: 0"],axis =1,inplace = True)
#     return df3
# def plot_df(df3):
#     fig = df3.plot().get_figure()
#     string = time.ctime().replace(" ","-").replace(":","-")
#     fig.savefig(string+'.png', dpi=300)
    
# df_rl = pd.read_csv("info_dfWed-Aug-31-23-31-26-2022.csv")
# df_twap = pd.read_csv("info_dfWed-Aug-31-23-36-23-2022.csv")
# df_rl, df_twap =  analyse_df(df_rl), analyse_df(df_twap)

# Index = list(set(list(df_rl.index)).intersection(set(list(df_twap.index))))
# result = pd.DataFrame((df_rl["Advantage"].loc[Index] - df_twap["Advantage"].loc[Index])/df_twap["Advantage"].loc[Index] * 100)
# result.plot()
# fig = result.plot().get_figure()
# string = time.ctime().replace(" ","-").replace(":","-")
# fig.savefig(string+'.png', dpi=300)

# plot_df(df_rl)
# plot_df(df_twap)






















