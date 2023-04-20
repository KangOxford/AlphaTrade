import time
import gym
from gym_exchange import Config
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from gym_exchange.trading_environment.base_env.base_env import BaseEnv
from gym_exchange.trading_environment.training_env.train_env import TrainEnv
from train import utils
import warnings; warnings.filterwarnings("ignore") # clear warnings
import wandb
# from wandb.integration.sb3 import WandbCallback
from train.sb3 import WandbCallback


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
    config = {
        "policy_type": "MlpLstmPolicy",
        "total_timesteps": int(1e8),
    }

    run = wandb.init(
        project="AlphaTrade",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        save_code=True,  # optional
    )


    def make_env():
        Config.train_env = "BaseEnv"
        env = Monitor(TrainEnv())  # record stats such as returns
        env = Monitor(env)  # record stats such as returns
        return env
    venv = DummyVecEnv([make_env])


    model = RecurrentPPO(
        config["policy_type"],
         venv,
         verbose=1,
         learning_rate=utils.linear_schedule(1e-3),
         tensorboard_log=f"{path}train/output/runs/{run.id}")

    model.learn(
       tb_log_name="RNN_PPO_Wandb",
        total_timesteps=config["total_timesteps"],
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{run.id}",
            verbose=1,
        ),
        log_interval=1,

    )

    run.finish()

