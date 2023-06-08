from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from gym_exchange.environment.base_env.base_env import BaseEnv

from train.sb3 import utils

import warnings; warnings.filterwarnings("ignore") # clear warnings

import wandb
# from wandb.integration.sb3 import WandbCallback
from train.sb3.sb3 import WandbCallback


if __name__ == "__main__":
    config = {
        "policy_type": "MlpLstmPolicy",
        "total_timesteps": int(1e8),
        # "env_name": "GymExchange-v1",
    }
    run = wandb.init(
        project="AlphaTrade",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )


    def make_env():
        # env = gym.make(config["env_name"])
        env = BaseEnv(),
        env = Monitor(env)  # record stats such as returns
        return env

    venv = DummyVecEnv([make_env])


    model = RecurrentPPO(config["policy_type"],
                         venv,
                         verbose=1,
                         learning_rate=utils.linear_schedule(1e-3),
                         tensorboard_log=f"/Users/kang/AlphaTrade/train/output/runs/{run.id}")

    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{run.id}",
            verbose=2,
        ),
    )

    # model = RecurrentPPO(
        # "MlpLstmPolicy",
        # venv,
        # verbose=1,
        # learning_rate=utils.linear_schedule(1e-3),
        # tensorboard_log="/Users/kang/AlphaTrade/train/output/")

    # model.learn(
    #     tb_log_name="RNN_PPO_initial",
    #     # eval_env = venv,
    #     callback=utils.TensorboardCallback()
    # )

    run.finish()

