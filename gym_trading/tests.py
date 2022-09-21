import numpy as np
import pandas as pd
from gym_trading.envs.broker import Flag
# =========================== RANDOM ======================================
def random_strategy(Env):
    import random
    random.seed( Flag.tests_seed )
    from gym_trading.data.data_pipeline import ExternalData
    from gym_trading.data.data_pipeline import Debug; Debug.if_return_single_flie = True # if False then return Flow_list # True if you want to debug
    Flow = ExternalData.get_sample_order_book_data()
    env = Env(Flow)
    obs = env.reset()
    diff_list = []
    step_list = []
    left_list = []
    Performance_list = []
    for i in range(int(1e8)):
        action = random.randint(0, Flag.max_action)
        observation, reward, done, info = env.step(action)
        env.render()
        if done:
            if str(Env) == "OptimalLiquidation":
                step_list.append(info['Step'])
                left_list.append(info['Left'])
            # breakpoint() # tbd
            env.reset()
    print(f"End of main(), Performance is {np.mean(Performance_list)}, Diff is {np.mean(diff_list)}, Step is {np.mean(step_list)}, Left is {np.mean(left_list)}")
# =============================================================================

# =========================== ZERO ======================================
def zero_strategy(Env):
    from gym_trading.data.data_pipeline import ExternalData
    from gym_trading.data.data_pipeline import Debug; Debug.if_return_single_flie = True # if False then return Flow_list # True if you want to debug
    Flow = ExternalData.get_sample_order_book_data()
    env = Env(Flow)
    obs = env.reset()
    action = 0
    diff_list = []
    step_list = []
    left_list = []
    Performance_list = []
    for i in range(int(1e8)):
        observation, reward, done, info = env.step(action)
        env.render()
        if done:
            if str(Env) == "OptimalLiquidation":
                step_list.append(info['Step'])
                left_list.append(info['Left'])
            env.reset()
    print(f"End of main(), Performance is {np.mean(Performance_list)}, Diff is {np.mean(diff_list)}, Step is {np.mean(step_list)}, Left is {np.mean(left_list)}")
# =============================================================================

# ======================   TWAP =======================================
def twap_strategy(Env):
    from gym_trading.data.data_pipeline import ExternalData
    from gym_trading.data.data_pipeline import Debug; Debug.if_return_single_flie = True # if False then return Flow_list # True if you want to debug
    Flow = ExternalData.get_sample_order_book_data()
    env = Env(Flow)
    obs = env.reset()
    # action = Flag.num2liquidate//Flag.max_episode_steps 
    action = Flag.num2liquidate//Flag.max_episode_steps + 1
    diff_list = []
    step_list = []
    left_list = []
    Performance_list = []
    for i in range(int(1e8)):
        observation, reward, done, info = env.step(action)
        # if i//2 == i/2: observation, reward, done, info = env.step(action)
        # if i//3 == i/3: observation, reward, done, info = env.step(action)
        # else: observation, reward, done, info = env.step(0)
        env.render()
        if done:
            if str(Env) == "OptimalLiquidation":
                step_list.append(info['Step'])
                left_list.append(info['Left'])
            env.reset()
    print(f"End of main(), Performance is {np.mean(Performance_list)}, Diff is {np.mean(diff_list)}, Step is {np.mean(step_list)}, Left is {np.mean(left_list)}")
# =============================================================================



if __name__=="__main__":
    pass