from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from gym_exchange.trading_environment.basic_env.base_env import BaseEnv

from train import utils

import warnings; warnings.filterwarnings("ignore") # clear warnings



if __name__ == "__main__":

    def make_env():
        # env = gym.make(config["env_name"])
        env = Monitor(BaseEnv())  # record stats such as returns
        return env

    venv = DummyVecEnv([make_env])


    model = RecurrentPPO(
        "MlpLstmPolicy",
        venv,
        verbose=1,
        learning_rate=utils.linear_schedule(1e-3),
        tensorboard_log="/Users/kang/AlphaTrade/train/output/")

    model.learn(
        tb_log_name="RNN_PPO_initial",
        total_timesteps = int(1e8),
        # eval_env = venv,
        callback=utils.TensorboardCallback()
    )


