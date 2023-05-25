from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from train.sb3 import utils
import warnings; warnings.filterwarnings("ignore") # clear warnings
import wandb
from train.sb3.sb3 import WandbCallback
import os
os.system("export PYTHONPATH=$PYTHONPATH:/home/duser/AlphaTrade")
path = utils.get_path_by_platform()
from gym_exchange.environment.timewindow_env.timewindow_env import TimewindowEnv

class TrainEnv(TimewindowEnv):

    # ========================== 03 ==========================
    def state(self, action):
        action[0] = 1 # 1 means sell stocks, 0 means buy stocks "Execution-3FreeDegrees"
        # action[2] = 0 # passive orders
        print(f"{action[0]} {action[1]} {action[2]}")  #$ less memory use
        ''''''
        action[1] = int(round(action[1]))
        action[2] = int(round(action[2]))
        ''''''
        state = super().state(action)
        return state
def main():
    # env = Monitor(TrainEnv())  # record stats such as returns
    # check_env(env)  # $

    config = {
        "policy_type": "MlpLstmPolicy",
        "total_timesteps": int(1e12),
        # "ent_coef" : 1,
        # ent_coef = 0.95,
        "ent_coef" : 0.5,
        # "ent_coef" : 0.1,
        # "ent_coef" : 0.01,
        "vf_coef" : 0.5,
        "gamma" : 1,  # Increase this value to make the agent more long-sighted
        # "gamma" : 0.9999999999,  # Increase this value to make the agent more long-sighted
        # "gae_lambda" : 0.999999,  # Lower this value to make the agent more sensitive to immediate rewards
        "gae_lambda" : 0.95,  # Lower this value to make the agent more sensitive to immediate rewards
        # "gae_lambda" : 0.8,  # Lower this value to make the agent more sensitive to immediate rewards
        # "gae_lambda" : 0.5,  # Lower this value to make the agent more sensitive to immediate rewards
        "clip_range" : 0.5,  # Increase this value to allow for more significant policy updates
    }

    run = wandb.init(
        project="Execution-2FreeDegrees",
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
        # learning_rate=utils.linear_schedule(5e-3),
        learning_rate = int(1e-4),
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