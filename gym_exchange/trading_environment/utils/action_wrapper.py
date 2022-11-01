# class ActionWrapper():
#     def __init__(self):
#         pass
#     def step(self):
#         pass
import numpy as np
from gym_exchange.trading_environment.action import Action
from gym_exchange.trading_environment.action import BaseAction
from gym_exchange.trading_environment.env_interface import SpaceParams


def action_wrapper(action: Action):
    price_delta = action.price_delta - SpaceParams.Action.price_delta_size_one_side
    side = 'bid' if action.side == 0 else 'ask'
    quantity = action.quantity - SpaceParams.Action.quantity_size_one_side
    new_action = BaseAction(side, quantity, price_delta)
    return new_action

def decoding(action: BaseAction):
    price_delta = action.price_delta + SpaceParams.price_delta_size_one_side
    side = 0 if action.side == 'bid' else 1
    quantity  = action.quantity + SpaceParams.Action.quantity_size_one_side
    result = [side, quantity, price_delta]
    wrapped_result = np.array(result)
    return wrapped_result

if __name__ == "__main__":
    action = Action(side = 'bid', quantity = 1, price_delta = 5)
    new_action = action_wrapper(action)
    print(new_action)