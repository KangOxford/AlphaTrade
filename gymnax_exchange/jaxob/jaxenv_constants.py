action_space= "spread_skew"
reward_space = "portfolio_value"
observation_space = "messages_new_tokenizer"
end_fn = "unwind_ref_price"
n_ticks_in_book =1
n_actions=8
reference_price_portfolio_value="best_bid_ask"
inv_penalty="none"
fixed_quant_value=10
asymmetrically_dampened_lambda=0.8
inventoryPnL_lambda=1.0
start_resolution=300  # 60*5, interval in seconds at which episodes start

num_messages_by_agent=4 #4 for "fixed_quant", n_actions*2 for other
