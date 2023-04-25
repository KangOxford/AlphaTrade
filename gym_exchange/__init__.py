import os
from os import listdir;from os.path import isfile, join

def get_symbol_date(AlphaTradeRoot):
    data_path = AlphaTradeRoot+"data"
    onlyfiles = sorted([f for f in listdir(data_path) if isfile(join(data_path, f))])
    for filename in onlyfiles:
        if filename != ".DS_Store":
            symbol = filename.split("_")[0]
            date = filename.split("_")[1]
            # Process the file here
            break
    return symbol, date

class Config:
    # --------------- 01 Basic ---------------
    # tick_size = 100 #(s hould be divided by 10000 to be converted to currency)
    price_level = 10
    lobster_scaling = 10000 # Dollar price times 10000 (i.e., A stock price of $91.14 is given by 911400)
    # max_horizon = 4096
    max_horizon = 2048
    # max_horizon = 1600
    # max_horizon = 800
    # max_horizon = 600



    # --------------- 00 Data ---------------
    # ············· 00.01 Window ············
    window_size = 100
    # ············· 00.01 Adapter ············
    raw_price_level = 10
    # raw_horizon = 2048
    # raw_horizon = 3700
    # raw_horizon = 4096
    raw_horizon = int(max_horizon * window_size * 1.01)
    # NOTE!!! need to be updated together with max_horizon
    # last position 1.763 used as redundant data
    type5_id_bid = 30000000  # caution about the volumn for valid numbers
    type5_id_ask = 40000000  # caution about the volumn for valid numbers
    # ············· 00.02 Source ············
    exchange_data_source = "raw_encoder"
    # exchange_data_source = "encoder"

    AlphaTradeRoot=os.path.join(os.path.dirname(__file__),'../')
    symbol, date = get_symbol_date(AlphaTradeRoot)
    # symbol = "TSLA";date = "2015-01-02"
    # symbol = "AMZN";date = "2021-04-01"




    # --------------- 02 Reward ---------------
    low_dimension_penalty_parameter = 1 # todo not sure
    cost_parameter = 5e-5
    # cost_parameter = 5e-6 # from paper.p29 : https://epubs.siam.org/doi/epdf/10.1137/20M1382386
    phi_prime = 5e-6 # from paper.p29 : https://epubs.siam.org/doi/epdf/10.1137/20M1382386
    # mu_regularity = 1
    # mu_regularity = 0.1
    # mu_regularity = 0.01
    mu_regularity = 0 # no peer reward, just revenue

    # --------------- 03 Task ---------------
    num2liquidate = 200 # 1 min
    '''num2liquidate = 2000 # 10 min, 200 # 1 min, 100 # 1/2 min'''

    # --------------- 04 Action ---------------
    # quantity_size_one_side = 30
    # quantity_size_one_side = 3
    # quantity_size_one_side = 1
    quantity_size_negative_side = 2
    quantity_size_positive_side = 200
    # quantity_size_positive_side = 100
    # quantity_size_positive_side = 30
    # quantity_size_positive_side = 20
    # quantity_size_positive_side = 10
    # quantity_size_positive_side = 2
    timeout = 100
    # timeout = 50
    # timeout = 10
    # timeout = 2
    # timeout = 1

    # --------------- 05 ActionWrapper --------
    trade_id_generator = 80000000
    order_id_generator = 88000000

    # --------------- 06 Space ---------------
    max_action = 300
    max_quantity = 3000 # TODO is it the same function with max_action?
    max_price = 35000000 # upper bound
    min_price = 30000000 # lower bound
    min_quantity = 0
    scaling = 30000000
    min_num_left = 0
    max_num_left = num2liquidate
    min_step_left= 0
    max_step_left = max_horizon
    state_dim_1 = 2 # price, quantity
    state_dim_2 = price_level # equals 10

    # --------------- 07 Observation ---------------
    # lock_back_window = 60 # all the states after 60 actions was conducted
    # num_ticks =  lock_back_window * skip # 1min, num_ticks
    '''num_ticks = 1200 # 1min, num_ticks'''

    # --------------- 08 Output ---------------
    out_path = '/Users/kang/AlphaTrade/gym_exchange/outputs/'

    # --------------- 09 Random ---------------
    seed = 1234
    # --------------- 10 TrainEnv ---------------
    # train_env = "BaseEnv"
    train_env = "BasicEnv"
    # --------------- 11 FillNa ---------------
    ask_fillna = max_price
    bid_fillna = min_price

    # --------------- 12 Benchmark ---------------
    sum_reward = 6257641200 # initial policy



from gym.envs.registration import register
register(
    id = "GymExchange-v1",
    # path to the class for creating the env
    # Note: entry_point also accept a class as input (and not only a string)
    entry_point="gym_exchange.environments.base_env.base_env:BaseEnv",
    # Max number of steps per episode, using a `TimeLimitWrapper`
    max_episode_steps=Config.max_horizon,
    )

