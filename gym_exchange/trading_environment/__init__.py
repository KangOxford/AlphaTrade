from gym.envs.registration import register

class Config:
    lobster_scaling = 10000 # Dollar price times 10000 (i.e., A stock price of $91.14 is given by 911400)
    max_episode_steps = 12000 # 10 mins
    # max_episode_steps= 1200 # 1 min
    # max_episode_steps= 600 # 1/2 min
    

    num2liquidate = 2000 # 10 min
    # num2liquidate = 200 # 1 min
    # num2liquidate = 100 # 1/2 min
    max_action = 300
    max_quantity = 3000 # TODO is it the same function with max_action?
    max_price = 35000000 # upper bound
    min_price = 30000000 # lower bound
    min_quantity = 0
    scaling = 30000000
    low_dimension_penalty_parameter = 1 # todo not sure
    cost_parameter = 5e-6 # from paper.p29 : https://epubs.siam.org/doi/epdf/10.1137/20M1382386
    # skip = 1 # 50 miliseconds
    # skip = 2 # default = 1 from step No.n to step No.n+1
    skip = 20 # 1 second 
    # skip = 200 # 10 seconds
    # skip = 1200 # 1 minute
    max_horizon = int(max_episode_steps / skip) # caution, need to be integer
    price_level = 10
    test_seed = 2022
    pretrain_steps = int(1e3)
    runing_penalty_parameter = 100
    time_window_size = 1
    min_num_left = 0
    max_num_left = num2liquidate
    min_step_left= 0
    max_step_left = max_episode_steps
    state_dim_1 = 2
    state_dim_2 = 10 
    # state_dim_2 = 12 # used to be 10
    state_dim_3 = time_window_size
    tick_size = 100 #(should be divided by 10000 to be converted to currency)

register(
    id = "GymExchange-v1",
    # path to the class for creating the env
    # Note: entry_point also accept a class as input (and not only a string)
    entry_point="gym_exchange.trading_environments.stock_env:StockEnv", # TODO
    kwargs={'Flow': True},
    # Max number of steps per episode, using a `TimeLimitWrapper`
    max_episode_steps=Config.max_episode_steps,
    )


