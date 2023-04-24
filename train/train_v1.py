from gym_exchange import Config
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from gym_exchange.environment.training_env.train_env import TrainEnv
from train import utils
import warnings; warnings.filterwarnings("ignore") # clear warnings
import wandb
from train.sb3 import WandbCallback
from stable_baselines3.common.env_checker import check_env
import os
os.system("export PYTHONPATH=$PYTHONPATH:/home/duser/AlphaTrade")
path = utils.get_path_by_platform()

def main():
    # env = Monitor(TrainEnv())  # record stats such as returns
    # check_env(env)  # $

    config = {
        "policy_type": "MlpLstmPolicy",
        "total_timesteps": int(1e12),
        "ent_coef" : 1,
        # ent_coef = 0.1,
        # ent_coef = 0.01,
        "vf_coef" : 0.5,
        "gamma" : 0.90,  # Lower this value to make the agent more short-sighted
        "gae_lambda" : 0.8,  # Lower this value to make the agent more sensitive to immediate rewards
        "clip_range" : 0.5,  # Increase this value to allow for more significant policy updates
    }

    run = wandb.init(
        project="AlphaTrade",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        save_code=True,  # optional
    )


    def make_env():
        env = TrainEnv()  # record stats such as returns
        env = Monitor(env)  # record stats such as returns
        return env
    # venv = DummyVecEnv([make_env] * 4)
    venv = DummyVecEnv([make_env] )


    model = RecurrentPPO(
        config["policy_type"],
        venv,
        ent_coef= config["ent_coef"],
        vf_coef= config["vf_coef"],
        gamma= config["gamma"],
        gae_lambda= config["gae_lambda"],
        clip_range= config["clip_range"],
        # env = venv,
        verbose=1,
        learning_rate=utils.linear_schedule(5e-3),
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

if __name__ == "__main__":
    main()
