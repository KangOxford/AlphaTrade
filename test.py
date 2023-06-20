import numpy as np
ask = np.load("/homes/80/kang/AlphaTrade/ask.npy")
bid = np.load("/homes/80/kang/AlphaTrade/bid.npy")

getBestAsksQtys = lambda x: x[:, np.argmin(np.where(x[:, :, 0] >= 0, x[:, :, 0], jnp.inf), axis=1), 1][:,0]
# getBestBidsQtys = lambda x: x[:, np.argmax(x[:, 0, :], axis=1), 1]
getBestBidsQtys = lambda x: x[:, np.argmax(x[:, :, 0], axis=0), 1]
# bestAsksQtys, bestBidsQtys = map(lambda func, orders: func(orders), [getBestAsksQtys, getBestBidsQtys], [state.ask_raw_orders, state.bid_raw_orders])
bestAsksQtys = getBestAsksQtys(ask)
bestBidsQtys = getBestBidsQtys(bid)
bestBidsQtys

lst= []
for arr in bid:
    lst.append(arr[np.argmax(arr[:,0]),1])
print(lst)






































