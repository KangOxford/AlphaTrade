import envs.gym_trading

# >>> 01.Initializes a trading environment.
environment = gym_trading.BaseEnvironment()

# >>> 02.Makes an initial observation.
observation = environment.reset()
done = False

while not done:
    