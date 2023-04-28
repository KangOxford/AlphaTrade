from ray.rllib.algorithms.ppo import PPOConfig
config = PPOConfig()
config = config.training(gamma=0.9, lr=0.01, kl_coeff=0.3)
config = config.resources(num_gpus=0)
config = config.rollouts(num_rollout_workers=4)
print(config.to_dict())
# Build a Algorithm object from the config and run 1 training iteration.
algo = config.build(env="GymExchange-v1")
algo.train()

