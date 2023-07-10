from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
# from train.sb3 import utils
import warnings; warnings.filterwarnings("ignore") # clear warnings
import wandb
import numpy as np
# from train.sb3.sb3 import WandbCallback
import os
os.system("export PYTHONPATH=$PYTHONPATH:/home/duser/AlphaTrade/")
os.system("export PYTHONPATH=$PYTHONPATH:/homes/80/kang/SpeedTesting/")
# path = utils.get_path_by_platform()

os.sys.path.append("/homes/80/kang/AlphaTrade/")
# os.sys.path.append("/homes/80/kang/SpeedTesting/")
# os.sys.path.append("/home/kanli/speedTesting/")
print(os.sys.path)

from pathlib import Path
home = str(Path.home())
path = home + "/AlphaTrade/"
# path = home + "/SpeedTesting/"
# path = home + "/speedTesting/"
import platform
if platform.system() == 'Darwin':
    print("Running on MacOS")
elif platform.system() == 'Linux':
    print("Running on Linux")
    import sys
    sys.path.append(path)
    sys.path.append(path + 'gym_exchange')
    sys.path.append(path + 'gymnax_exchange')
else:
    print("Unknown operating system")
print(os.sys.path)

from gym_exchange import Config

Config.train_env = "BaseEnv"
print(f"Config.train_env: {Config.train_env}")

from gym_exchange.environment.jaxLike_env.jaxLike_env import JaxLikeEnv



# ============load data==============
from gym_exchange.data_orderbook_adapter.raw_encoder import RawDecoder,RawEncoder
from gym_exchange.data_orderbook_adapter.decoder import Decoder
from gym_exchange.data_orderbook_adapter.encoder import Encoder
from gym_exchange.data_orderbook_adapter.data_pipeline import DataPipeline
def to_order_flow_lists(flow_lists):
    '''change side format from bid/ask to 1/-1
    side = -1 if item.side == 'ask' else 1'''
    for flow_list in flow_lists:
        for item in flow_list:
            side = -1 if item.side == 'ask' else 1
            item.side = side
    return flow_lists
def flow_lists_initialization():
    if Config.exchange_data_source == "encoder":
        decoder = Decoder(**DataPipeline()())
        encoder = Encoder(decoder)
    elif Config.exchange_data_source == "raw_encoder":
        decoder = RawDecoder(**DataPipeline()())
        encoder = RawEncoder(decoder)
    else:
        raise NotImplementedError
    flow_lists = encoder()
    flow_lists = to_order_flow_lists(flow_lists)
    return flow_lists

flow_lists_initialized = flow_lists_initialization()
# ============load data==============
class TrainEnv(JaxLikeEnv):

    # ========================== 03 ==========================
    def state(self, action):
        '''
        action[0] = 1 # 1 means sell stocks, 0 means buy stocks "Execution-3FreeDegrees"
        # action[2] = 0 # passive orders
        action[1] = int(round(action[1]))
        action[2] = int(round(action[2]))
        '''
        state = super().state(action)
        return state
def main():
    # env = Monitor(TrainEnv())  # record stats such as returns
    # check_env(env)  # $
    import time
    start = time.time()

    play_around=True
    train=False

    config = {
        "policy_type": "MlpLstmPolicy",
        "total_timesteps": int(50000),
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

    def make_env():
            env = TrainEnv(flow_lists_initialized)  # record stats such as returns
            env = Monitor(env)  # record stats such as returns
            return env
    

    if play_around:
        test_env=make_env()
        test_env.reset()
        test_env.step(np.array([10,12,0,0]))

        print('hello there')


    if train:
        run = wandb.init(
        project="TestingCPU",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        save_code=True,  # optional
        )

        
        venv = DummyVecEnv([make_env] * 1000) # <= <= <= <= <= <= Your can tune this
        # venv = DummyVecEnv([make_env] )


        model = RecurrentPPO(
            config["policy_type"],
            venv,
            ent_coef= config["ent_coef"],
            vf_coef= config["vf_coef"],
            gamma= config["gamma"],
            gae_lambda= config["gae_lambda"],
            clip_range= config["clip_range"],


            batch_size=2500, # <= <= <= <= <= <= Your can tune this
            n_steps=10, # <= <= <= <= <= <= Your can tune this

            n_epochs=4, # <= <= <= <= <= <= Your can tune this
            verbose=1, 
            # learning_rate=utils.linear_schedule(5e-3),
            learning_rate = int(1e-4),
            tensorboard_log=f"{path}train/output/runs/{run.id}"
        )


    
        start = time.time()
        model.learn(
            tb_log_name="RNN_PPO_Wandb",
            total_timesteps=config["total_timesteps"],
            #callback=WandbCallback(
            #    gradient_save_freq=100,
            #    model_save_path=f"models/{run.id}",
            #    verbose=1,
            # ),
            log_interval=1,
        )
        end = time.time()
        print(f"*** main starts at {start}, and ends at {end}, takes {end-start}.")
        #run.finish()

if __name__ == "__main__":
    main()
