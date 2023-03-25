# ======================================= plot ============================
import pandas as pd
import matplotlib.pyplot as plt
from gym_exchange import Config

def plot_render(self):
    # ----------------------- func ---------------------
    def split_market_step_vwap(market_step_vwap):
        ask_market_step_vwap = {}
        bid_market_step_vwap = {}
        for key, value in market_step_vwap.items():
            if value > mid_prices.get(key):
                ask_market_step_vwap[key] = value
            elif value < mid_prices.get(key):
                bid_market_step_vwap[key] = value
            else:
                raise NotImplementedError
        return ask_market_step_vwap, bid_market_step_vwap
    # ----------------------- date ---------------------
    string = "[TASK]optimal acquisition  [AIM]As lowest executed prices(green in the fig) as possible"
    # string = "[ACTION]Action(direction = 'bid', quantity_delta = 0, price_delta = 0), [AIM]As lowest prices as possible"
    # recorder = env.exchange.executed_pairs_recoder
    recorder = self.exchange.executed_pairs_recoder
    agent_pairs = recorder.agent_pairs
    market_pairs = recorder.market_pairs
    from gym_exchange.trading_environment.basic_env.utils import vwap_price

    agent_step_vwap = {k: vwap_price(v) for k, v in agent_pairs.items()}
    market_step_vwap = {k: vwap_price(v) for k, v in market_pairs.items()}
    mid_prices = {k: v for k, v in enumerate(self.exchange.mid_prices)}
    best_bids = {k: v for k, v in enumerate(self.exchange.best_bids)}
    best_asks = {k: v for k, v in enumerate(self.exchange.best_asks)}
    # ----------------------- fig ---------------------
    fig, ax = plt.subplots(figsize=(25, 10))
    ask_market_step_vwap, bid_market_step_vwap = split_market_step_vwap(market_step_vwap)
    ax.plot(pd.Series(best_asks), label = "best_ask", color = 'dodgerblue', linewidth=0.5)
    ax.plot(pd.Series(mid_prices), label="mid_price", color='darkorange', linestyle = ":", dashes=[1, 0.5], linewidth=1.0)
    ax.plot(pd.Series(best_bids), label="best_bid", color = 'red', linewidth=0.5)
    ax.scatter(ask_market_step_vwap.keys(), ask_market_step_vwap.values(), label="ask_market_step_vwap", color='royalblue', s = 1)
    ax.scatter(bid_market_step_vwap.keys(), bid_market_step_vwap.values(), label="bid_market_step_vwap", color='deeppink', s = 1)
    ax.scatter(agent_step_vwap.keys(), agent_step_vwap.values(), label='Agent', color='lime')
    plt.legend()
    plt.xlabel("index of the time step(with a time delta of 10 ms)" + "\n\n" + \
               f'max_horizon: {Config.max_horizon}' + "\n" + \
               f'num_to_liquidate: {Config.num2liquidate}')
               # f"actual_action: ({self.info['Actual_action/Price']}: {self.info['Actual_action/Quantity']})")
    plt.ylabel("price of the stcok AMZN on 2021-04-01")
    plt.title(string)
    plt.subplots_adjust(left=0.05, right=0.95)
    # plt.subplots_adjust(left=0.05, right=0.95, bottom=0.20, top=0.80)
    import datetime; timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    fig_string = f'output_{timestamp}.png'
    plt.savefig(Config.out_path + fig_string)
    plt.show()
