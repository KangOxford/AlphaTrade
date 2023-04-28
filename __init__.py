from gym.envs.registration import register
from gym_exchange import Config
register(
    id = "GymExchange-v1",
    # id = "GymExchange-v1",
    # path to the class for creating the env
    # Note: entry_point also accept a class as input (and not only a string)
    # entry_point="gym_exchange.environments.base_env.base_env:BaseEnv",
    entry_point="gym_exchange.environment.training_env.train_env:TrainEnv",
    # Max number of steps per episode, using a `TimeLimitWrapper`
    max_episode_steps=Config.max_horizon,
    )
