# import numpy as np
# ask = np.load("/homes/80/kang/AlphaTrade/ask.npy")
# bid = np.load("/homes/80/kang/AlphaTrade/bid.npy")

# getBestAsksQtys = lambda x: x[:, np.argmin(np.where(x[:, :, 0] >= 0, x[:, :, 0], jnp.inf), axis=1), 1][:,0]
# # getBestBidsQtys = lambda x: x[:, np.argmax(x[:, 0, :], axis=1), 1]
# getBestBidsQtys = lambda x: x[:, np.argmax(x[:, :, 0], axis=0), 1]
# # bestAsksQtys, bestBidsQtys = map(lambda func, orders: func(orders), [getBestAsksQtys, getBestBidsQtys], [state.ask_raw_orders, state.bid_raw_orders])
# bestAsksQtys = getBestAsksQtys(ask)
# bestBidsQtys = getBestBidsQtys(bid)
# bestBidsQtys

# lst= []
# for arr in bid:
#     lst.append(arr[np.argmax(arr[:,0]),1])
# print(lst)

# # ================================================================================
import jax
import jax.numpy as jnp
executed_ = jnp.array(
      [[ 31272500,        20,     -9001,     -8999,     43200,   1922666],
       [ 31271100,         1,     -9003, 201017109,     43201, 432381457],
       [ 31269600,        14,     -9005, 201017109,     43201, 432381457],
       [ 31268700,         1,     -9007, 201017109,     43201, 432381457],
       [ 31267900,         1,     -9009, 201017109,     43201, 432381457],
       [ 31267000,         1, 201446565, 201017109,     43201, 432381457],
       [ 31267000,         1, 201447253, 201017109,     43201, 432381457],
       [ 31266900,         1,     -9011, 197160893,     43201, 432413666],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0]])

# # a = jnp.array([0, 1, 0, 0, 0, 0])
# # b = jnp.ones(100,dtype=jnp.int32).T
# # @jax.jit
# # def get1():
# #     E = executed.T @ executed
# #     e = executed @ a
# #     esum= e @ b
# #     vwap = E[0, 1] / esum
# #     return vwap
# # @jax.jit
# # def get2():
# #     E = executed.T @ executed
# #     e = executed @ a
# #     vwap = E[0, 1] / e.sum()
# #     return vwap
# # @jax.jit
# # def get3():
# #     vwap = (executed[:, 0] * executed[:, 1]).sum() / executed[:1].sum()
# #     return vwap
# # import jax
# # import jax.numpy as jnp
# # import timeit


# # execution_time0 = timeit.timeit(lambda: jax.device_get(get1().block_until_ready()), number=10)
# # # print("Execution time for get1():", execution_time, "seconds")

# # # Measure the execution time for get2()
# # execution_time0 = timeit.timeit(lambda: jax.device_get(get2().block_until_ready()), number=10)
# # # print("Execution time for get2():", execution_time, "seconds")

# # # Measure the execution time for get3()
# # execution_time0 = timeit.timeit(lambda: jax.device_get(get3().block_until_ready()), number=10)


# # # Measure the execution time for get1()
# # execution_time = timeit.timeit(lambda: jax.device_get(get1().block_until_ready()), number=100000)
# # print("Execution time for get1():", execution_time, "seconds")

# # # Measure the execution time for get2()
# # execution_time = timeit.timeit(lambda: jax.device_get(get2().block_until_ready()), number=100000)
# # print("Execution time for get2():", execution_time, "seconds")

# # # Measure the execution time for get3()
# # execution_time = timeit.timeit(lambda: jax.device_get(get3().block_until_ready()), number=100000)
# # print("Execution time for get3():", execution_time, "seconds")
# # # # Perform the computation and measure the time


# # ================================================================================

# import jax
# import jax.numpy as jnp
# import timeit


# @jax.jit
# def get_agent1():
#     e = jnp.sign(executed)
#     e1 = e-1
#     e2 = jnp.sign(-1 * jnp.bitwise_and(e1 , e) )
#     e3=jnp.multiply(e2, executed)
#     e4= jnp.sign(jnp.bitwise_and(e3,  jnp.sign(e3 + 9000)))
#     e5 = jnp.bitwise_and(e4,jnp.sign(1+e4))
#     return e5
#     # e6 = jnp.any(e5 == 1, axis=1)[:, jnp.newaxis]
#     # e7 = e6*executed
#     # return e7

# @jax.jit
# def get_agent2():
#     mask2 = ((-9000 < executed[:, 2]) & (executed[:, 2] < 0)) | ((-9000 < executed[:, 3]) & (executed[:, 3] < 0))
#     agentTrades = jnp.where(mask2[:, jnp.newaxis], executed, 0)
#     return agentTrades

# execution_time0 = timeit.timeit(lambda: jax.device_get(get_agent1().block_until_ready()), number=1)
# execution_time0 = timeit.timeit(lambda: jax.device_get(get_agent2().block_until_ready()), number=1)



# execution_time0 = timeit.timeit(lambda: jax.device_get(get_agent1().block_until_ready()), number=1)
# print("Execution time for get1():", execution_time0, "seconds")

# # Measure the execution time for get2()
# execution_time0 = timeit.timeit(lambda: jax.device_get(get_agent2().block_until_ready()), number=1)
# print("Execution time for get2():", execution_time0, "seconds")



# # Measure the execution time for get1()
# execution_time = timeit.timeit(lambda: jax.device_get(get_agent1().block_until_ready()), number=100000)
# print("Execution time for get1():", execution_time, "seconds")

# # Measure the execution time for get2()
# execution_time = timeit.timeit(lambda: jax.device_get(get_agent2().block_until_ready()), number=100000)
# print("Execution time for get2():", execution_time, "seconds")
# # ================================================================================

import jax
import jax.numpy as jnp
import timeit

trades = jnp.array([
       [ 31272500,        20,     -9001,     -8999,     43200,   1922666],
       [ 31271100,         1,     -9003, 201017109,     43201, 432381457],
       [ 31269600,        14,     -9005, 201017109,     43201, 432381457],
       [ 31268700,         1,     -9007, 201017109,     43201, 432381457],
       [ 31267900,         1,     -9009, 201017109,     43201, 432381457],
       [ 31267000,         1, 201446565, 201017109,     43201, 432381457],
       [ 31267000,         1, 201447253, 201017109,     43201, 432381457],
       [ 31266900,         1,     -9011, 197160893,     43201, 432413666],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1],
       [       -1,        -1,        -1,        -1,        -1,        -1]])

agentTrades_ = jnp.array(
[      [31272500,       20,    -9001,    -8999,    43200,  1922666],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0],
       [       0,        0,        0,        0,        0,        0]])
rewardValue_ = jnp.array(2.6544957e+08, dtype=jnp.float32)
# # reward=self.get_reward(state, params)
# executed = jnp.where((trades[:, 0] > 0)[:, jnp.newaxis], trades, 0)
# mask2 = ((-9000 < executed[:, 2]) & (executed[:, 2] < 0)) | ((-9000 < executed[:, 3]) & (executed[:, 3] < 0))
# agentTrades = jnp.where(mask2[:, jnp.newaxis], executed, 0)
# vwap = (executed[:,0] * executed[:,1]).sum()/ executed[:1].sum() 
# advantage = (agentTrades[:,0] * agentTrades[:,1]).sum() - vwap * agentTrades[:,1].sum()
# Lambda = 0.5 # FIXME shoud be moved to EnvState or EnvParams
# drift = agentTrades[:,1].sum() * (vwap - 36000000)
# rewardValue = advantage + Lambda * drift   
# reward = jnp.sign(agentTrades[0,0]) * rewardValue # if no value agentTrades then the reward is set to be zero
# reward=jnp.nan_to_num(reward) 

# a = jnp.array([0, 1, 0, 0, 0, 0])
# b = jnp.ones(100,dtype=jnp.int32).T


# @jax.jit
# def get1(trades):
#     # reward=self.get_reward(state, params)
#     executed = jnp.where((trades[:, 0] > 0)[:, jnp.newaxis], trades, 0)
#     mask2 = ((-9000 < executed[:, 2]) & (executed[:, 2] < 0)) | ((-9000 < executed[:, 3]) & (executed[:, 3] < 0))
#     agentTrades = jnp.where(mask2[:, jnp.newaxis], executed, 0)
#     return agentTrades
# @jax.jit
# def get2(executed, agentTrades):
#     E = executed.T @ executed
#     e = executed @ a
#     esum= e @ b
#     vwap = E[0, 1] / esum
#     # vwap = (executed[:,0] * executed[:,1]).sum()/ executed[:1].sum() 
#     advantage = (agentTrades[:,0] * agentTrades[:,1]).sum() - vwap * agentTrades[:,1].sum()
#     Lambda = 0.5 # FIXME shoud be moved to EnvState or EnvParams
#     drift = agentTrades[:,1].sum() * (vwap - 36000000)
#     rewardValue = advantage + Lambda * drift
#     return rewardValue
# @jax.jit
# def get3(agentTrades, rewardValue):
#     reward = jnp.sign(agentTrades[0,0]) * rewardValue # if no value agentTrades then the reward is set to be zero
#     reward=jnp.nan_to_num(reward)
#     return reward

# @jax.jit
# def get4(trades):
#     return get3(agentTrades_,get2(executed_, get1(trades)))

# # execution_time0 = timeit.timeit(lambda: jax.device_get(get1(trades).block_until_ready()), number=1)
# # execution_time0 = timeit.timeit(lambda: jax.device_get(get2(executed_, agentTrades_).block_until_ready()), number=1)
# # execution_time0 = timeit.timeit(lambda: jax.device_get(get3(agentTrades_, rewardValue_).block_until_ready()), number=1)
# execution_time0 = timeit.timeit(lambda: jax.device_get(get4(trades).block_until_ready()), number=1)


# # execution_time0 = timeit.timeit(lambda: jax.device_get(get1(trades).block_until_ready()), number=1)
# # print("Execution time for get1():", execution_time0, "seconds")
# # execution_time0 = timeit.timeit(lambda: jax.device_get(get2(executed_, agentTrades_).block_until_ready()), number=1)
# # print("Execution time for get2():", execution_time0, "seconds")
# # execution_time0 = timeit.timeit(lambda: jax.device_get(get3(agentTrades_, rewardValue_).block_until_ready()), number=1)
# # print("Execution time for get3():", execution_time0, "seconds")
# execution_time0 = timeit.timeit(lambda: jax.device_get(get4(trades).block_until_ready()), number=1)
# print("Execution time for get4():", execution_time0, "seconds")


# # #------------------------ testing ------------------------
# # # Measure the execution time for get1()
# # execution_time = timeit.timeit(lambda: jax.device_get(get1(trades).block_until_ready()), number=100000)
# # print("Execution time for get1():", execution_time, "seconds")

# # # Measure the execution time for get2()
# # execution_time = timeit.timeit(lambda: jax.device_get(get2(executed_, agentTrades_).block_until_ready()), number=100000)
# # print("Execution time for get2():", execution_time, "seconds")

# # # Measure the execution time for get3()
# # execution_time = timeit.timeit(lambda: jax.device_get(get3(agentTrades_, rewardValue_).block_until_ready()), number=100000)
# # print("Execution time for get3():", execution_time, "seconds")
# # # # Perform the computation and measure the time

# # # Measure the execution time for get3()
# # execution_time = timeit.timeit(lambda: jax.device_get(get4(trades).block_until_ready()), number=100000)
# # print("Execution time for get4():", execution_time, "seconds")
# # # # Perform the computation and measure the time


# # ================================================================================

trades_big = jnp.tile(trades,(1000,1))

@jax.jit
def get1(trades):
    # reward=self.get_reward(state, params)
    executed = jnp.where((trades[:, 0] > 0)[:, jnp.newaxis], trades, 0)
    mask2 = ((-9000 < executed[:, 2]) & (executed[:, 2] < 0)) | ((-9000 < executed[:, 3]) & (executed[:, 3] < 0))
    agentTrades = jnp.where(mask2[:, jnp.newaxis], executed, 0)
    return agentTrades

execution_time0 = timeit.timeit(lambda: jax.device_get(get1(trades).block_until_ready()), number=1)



execution_time0 = timeit.timeit(lambda: jax.device_get(get1(trades).block_until_ready()), number=1)
print("Execution time for get1_sml():", execution_time0, "seconds")
execution_time0 = timeit.timeit(lambda: jax.device_get(get1(trades_big).block_until_ready()), number=1)
print("Execution time for get1_big():", execution_time0, "seconds")


# #------------------------ testing ------------------------
# Measure the execution time for get1()
execution_time = timeit.timeit(lambda: jax.device_get(get1(trades).block_until_ready()), number=100*1000)
print("Execution time for get1_sml():", execution_time, "seconds")

# Measure the execution time for get2()
execution_time = timeit.timeit(lambda: jax.device_get(get1(trades_big).block_until_ready()), number=100*1000)
print("Execution time for get1_big():", execution_time, "seconds")

