from gym.envs.registration import register
from gym_trading.envs.broker import Flag

register(
    id = "GymTrading-v1",
    # path to the class for creating the env
    # Note: entry_point also accept a class as input (and not only a string)
    entry_point="gym_trading.envs.base_environment:BaseEnv",
    kwargs={'Flow': True},
    # Max number of steps per episode, using a `TimeLimitWrapper`
    max_episode_steps=Flag.max_episode_steps,
    )