from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
# from gym_exchange.trading_environment.basic_env.basic_env import BasicEnv
# from gym_exchange.trading_environment.base_env.base_env import BaseEnv
from gym_exchange.trading_environment.training_env.train_env import TrainEnv

from train import utils

import warnings; warnings.filterwarnings("ignore") # clear warnings

#System and standard inputs
import platform
if platform.system() == 'Darwin':
    print("Running on MacOS")
    path = "/Users/kang/AlphaTrade/"
elif platform.system() == 'Linux':
    print("Running on Linux")
    import sys
    path = "/home/kanli/AlphaTrade/"
    sys.path.append(path)
    sys.path.append(path + 'gym_exchange')
    sys.path.append(path + 'gymnax_exchange')
else:print("Unknown operating system")


if __name__ == "__main__":

    def make_env():
        # env = gym.make(config["env_name"])
        env = Monitor(TrainEnv())  # record stats such as returns
        return env

    # venv = DummyVecEnv([make_env ] * 4)
    venv = DummyVecEnv([make_env])


    model = RecurrentPPO(
        "MlpLstmPolicy",
        venv,
        verbose=1,
        learning_rate=utils.linear_schedule(1e-3),
        tensorboard_log=path + "train/output/")

    model.learn(
        tb_log_name="RNN_PPO",
        total_timesteps = int(1e8),
        callback=utils.TensorboardCallback(),
        progress_bar = True,
        log_interval = 1
        # eval_env = venv,
    )


