action_space= "fixed_quants"
reward_space = "zero_inv"
observation_space = "engineered"
end_fn = "unwind_mid_price"
n_ticks_in_book =1
n_actions=8
reference_price_portfolio_value="best_bid_ask"
inv_penalty="none"


num_messages_by_agent=4 #4 for "fixed_quant", n_actions*2 for other
