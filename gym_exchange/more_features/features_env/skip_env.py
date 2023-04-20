from gym_exchange import Config
from gym_exchange.more_features.features_env.memory_env import MemoEnv
from gym_exchange.environment.base_env.assets import Action
# from typing import TypeVar
# Action = TypeVar("Action")
# State = TypeVar("State")


# *************************** 4 *************************** #
class SkipEnv(MemoEnv):
    '''for action/step'''
    # ========================== 01 ==========================
    def __init__(self):
        super(SkipEnv, self).__init__()
    
    # ========================== 03 ==========================
    def step(self, action):
        '''input : action
           return: observation, reward, done, info
           during all the first window_size -1 steps, action is none
           then action is the params passed in at final step'''
        # ···················· 03.00.01 ···················· 
        for i in range(Config.skip -1):
            super().state(action = None) # no accumulator
            # state, reward, done, info = super().state(action), super().reward, super().done, super().info
        # ···················· 03.00.02 ···················· 
        state, reward, done, info = super().step(action = action) # accumulator called only once
        return state, reward, done, info
    # --------------------- 03.01 ---------------------

    @property
    def reward(self):
        self.reward_generator.update(self.exchange.executed_pairs_recoder.market_agent_executed_pairs_in_last_step, self.exchange.mid_prices[-1])
        reward = self.reward_generator.step()
        return reward
    # --------------------- 03.03  ---------------------
    @property
    def done(self):
        if self.num_left_processor.num_left <= 0 or self.cur_step >= Config.max_horizon : return True
        else : return False
        
if __name__ == "__main__":
    # --------------------- 05.01 --------------------- 
    # from stable_baselines3.common.env_checker import check_env
    # env = SkipEnv()
    # check_env(env)
    # --------------------- 05.02 --------------------- 
    env = SkipEnv()
    env.reset()
    for i in range(int(1e6)):
        # breakpoint()#$
        action = Action(side = 'bid', quantity = 1, price_delta = 1)
        state, reward, done, info = env.step(action.to_array)
        env.render()
        if done:
            env.reset()
            break #$























