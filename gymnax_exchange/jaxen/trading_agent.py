from gymnax_exchange.jaxen.StatesandParams import ExecEnvState, ExecEnvParams, MultiAgentState, WorldState

"""Incomplete TradingAgent class to be implemented by specific trading agents.
    Need to finish writing this, but will serve as a good sanity check."""


class TradingAgent:
    def __init__(self):
        raise NotImplementedError("This class must be inherited and implemented by a subclass, it serves as a template for trading agents.")
    
    def reset_env(self):
        raise NotImplementedError("This method must be implemented by the subclass to reset the trading environment.")
    
    def is_terminal(self, world_state: WorldState, agent_state: ExecEnvState):
        raise NotImplementedError("This method must be implemented by the subclass to check if