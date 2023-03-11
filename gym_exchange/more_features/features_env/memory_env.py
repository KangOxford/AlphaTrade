from gym_exchange.trading_environment.basic_env.assets import Action

# from gym_exchange.trading_environment.utils.action_wrapper import action_wrapper
from gym_exchange.trading_environment.basic_env.interface_env import State # types
from gym_exchange.trading_environment.basic_env.base_env import BaseEnv


# *************************** 3 *************************** #
class MemoEnv(BaseEnv):
    # ========================== 01 ==========================
    def __init__(self):
        super().__init__()
        
    # ========================== 02 ==========================
    '''for reset'''
    def initial_state(self) -> State:
        state = super().initial_state()
        self.state_memos = []
        return state
        
    # ========================== 03 ==========================
    '''for step'''
    # --------------------- 03.01 ---------------------
    def state(self, action: Action) -> State:
        state = super().state(action)
        self.state_memos.append(state)
        return state
    
    # ========================== 04 ==========================
    '''for render'''
    def render(self, mode = 'human'):
        super().render()
        
if __name__ == "__main__":
    # --------------------- 05.01 --------------------- 
    # from stable_baselines3.common.env_checker import check_env
    # env = MemoEnv()
    # check_env(env)
    # print("++++ Finish Checking the Environment")
    # import time; time.sleep(5)
    # --------------------- 05.02 --------------------- 
    env = MemoEnv()
    env.reset()
    # print("++++ Finish Reseting the Environment");import time; time.sleep(5)
    # breakpoint()#$
    for i in range(int(1e6)):
        # print("-"*20) #$
        # action = None
        action = Action(side = 'bid', quantity = 1, price_delta = 1)
        # print(action) #$
        # breakpoint() #$
        state, reward, done, info = env.step(action)
        # print(state) #$
        # print(reward) #$
        # print(done) #$
        # print(info) #$
        env.render()
        if done:
            print("++++ Encounter Done");import time;time.sleep(5)
            # env.reset();print("++++ Finish Reseting the Environment");import time;time.sleep(5)
            break #$
    # --------------------- 05.02 --------------------- 
    # env = MemoEnv()
    # env.reset()
    # for i in range(int(1e6)):
    #     print("-"*20) #$
    #     action = Action(side = 'bid', quantity = 1, price_delta = 1)
    #     print(action) #$
    #     # breakpoint() #$
    #     state, reward, done, info = env.step(action.to_array)
    #     print(state) #$
    #     print(reward) #$
    #     print(done) #$
    #     print(info) #$
    #     env.render()
    #     if done:
    #         env.reset()
    #         break #$
