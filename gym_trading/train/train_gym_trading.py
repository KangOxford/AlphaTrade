# %% 
# =============================================================================
import torch
# import torch.nn as nn
# import torch.nn.functional as F
# ----------------------------------------------------------------------------
from gym_trading.envs import base_environment
# =============================================================================

max_episode = int(1e6)


# >>> 01.Initializes a trading environment.
env = base_environment.BaseEnv()

for i_episode in range(1, max_episode + 1):
    episode_reward = 0
    observation = env.reset()
    running_reward = torch.tensor(0.0)
    for t in range(env.num_steps):
        pass
