import argparse

# >>> 01 Iintailize ArgumentParser <<<
# >>> 01.01 Iintailize Training Basics <<<
parser = argparse.ArgumentParser()
parser.add_argument("--epochs_number", type = int, default = 100)
parser.add_argument("--batch_size", type = int, default = 32) # the default for the most of choices.
# >>> 01.02 Iintailize Adam Parameters <<<
parser.add_argument("--learning_rate", type = float, default = 0.01)
parser.add_argument("--beta_firt_moment", type = float, default = 0.5)
parser.add_argument("--beta_second_moment", type = float, default = 0.99)
# >>> 01.03 Iintailize Network Parameters <<<
parser.add_argument("--latent_dim", type = int, default = 64)
parser.add_argument("--generated_dim", type = int, default = 32)
parser.add_argument("--channel_dim", type = int, default = 3)
# >>> 01.04 Parsing Parameters <<<
(options, args) = parser.parse_args()
print(options, args)