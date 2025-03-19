# Need only to add deviations from the default environment config.
mm_env_config_hps = []

for o in ["engineered"]:
    for r in ["pnl", "complex", "portfolio_value"]:
        for i in ["none", "linear", "quadratic"]:
            for a in ["fixed_prices", "fixed_quants"]:
                for e in ["force_market_order", "unwind_mid_price","do_nothing"]:
                    for r in ["mid", "best_bid_ask"]:
                        mm_env_config_hps.append({"observation_space":o,
                                                  "reward_space":r,
                                                  "inv_penalty":i,
                                                  "action_space":a,
                                                   "end_fn":e,
                                                   "reference_price_portfolio_value":r})  

# Model & Training parameters, should be independant of the environment config
# TODO: Some adjustment needed, some of these are effectively environment parameters
rnn_training_parameters = {
    "LR": {"values": [2.5e-4]},
    "NUM_ENVS": {"values": [256]},
    "NUM_STEPS": {"values": [128]},
    "TOTAL_TIMESTEPS": {"values": [4e5]},
    "UPDATE_EPOCHS": {"values": [4]},
    "NUM_MINIBATCHES": {"values": [16]},
    "GAMMA": {"values": [0.99]},
    "GAE_LAMBDA": {"values": [0.95]},
    "CLIP_EPS": {"values": [0.2]},
    "ENT_COEF": {"values": [0.0]},
    "VF_COEF": {"values": [0.5]},
    "MAX_GRAD_NORM": {"values": [0.5]},
    "ENV_NAME": {"values": ["AlphaTradeMM"]},
    "ANNEAL_LR": {"values": [True]},
    "DEBUG": {"values": [True]},
    "VERBOSE": {"values": [False]},
    "REWARD_LAMBDA": {"values": [0.1]},
    "ACTION_TYPE": {"values": ["pure"]},
    "WINDOW_INDEX": {"values": [200]},
    "MAX_TASK_SIZE": {"values": [100]},
    "EPISODE_TIME": {"values": [60*5]},
    "DATA_TYPE": {"values": ["fixed_time"]},
}


acs5_training_parameters = {
    "LR": {"values": [2.5e-4]},
    "NUM_ENVS": {"values": [1]},
    "NUM_STEPS": {"values": [128]},
    "TOTAL_TIMESTEPS": {"values": [2e6]},
    "UPDATE_EPOCHS": {"values": [4]},
    "NUM_MINIBATCHES": {"values": [1]},
    "GAMMA": {"values": [0.99]},
    "GAE_LAMBDA": {"values": [0.95]},
    "CLIP_EPS": {"values": [0.2]},
    "ENT_COEF": {"values": [0.01]},
    "VF_COEF": {"values": [0.5]},
    "MAX_GRAD_NORM": {"values": [0.5]},
    "ACTIVATION": {"values": ["tanh"]},
    "ANNEAL_LR": {"values": [True]},
    "DEBUG": {"values": [True]},
    "ENV_NAME": {"values": ["alphatradeExec-v0"]},
    "WINDOW_INDEX": {"values": [200]},
    "RNN_TYPE": {"values": ["S5"]},
    "HIDDEN_SIZE": {"values": [64]},
    "ACTIVATION_FN": {"values": ["relu"]},
    "ACTION_NOISE_COLOR": {"values": [2]},
    "TASKSIDE": {"values": ["random"]},
    "REWARD_LAMBDA": {"values": [1.0]},
    "ACTION_TYPE": {"values": ["pure"]},
    "OUT_SIZE": {"values": [2]},
    "CONT_ACTIONS": {"values": [False]},
    "JOINT_ACTOR_CRITIC_NET": {"values": [True]},
    "EPISODE_TIME": {"values": [60 * 5]},
    "EP_TYPE": {"values": ["fixed_time"]},
    "ACTOR_STD": {"values": ["state_dependent"]},
    "REDUCE_ACTION_SPACE_BY": {"values": [10]},
    "DATA_TYPE": {"values": ["fixed_time"]},

}
