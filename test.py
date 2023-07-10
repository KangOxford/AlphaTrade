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



# # ================================================================================

import jax
import jax.numpy as jnp

done = jnp.array([False, True, False])
done = jnp.array(
    [[False] * 510 , [True] * 510, [False] * 510]
)
obs_re = jnp.ones((3,510))
obs_st = jnp.zeros((3,510))
jax.lax.select(done, obs_re, obs_st)


import jax.numpy as jnp
import jax
agentTrades = \
jnp.array([[ 31272500,         9,     -8999,     -8999,     43200,   1922666],
       [ 31278200,        13,     -8998,     -8998,     43200,   1922666],
       [ 31279400,         1, 201543641,     -8997,     43200,   1922666],
       [ 31279400,         1, 201543925,     -8996,     43200,   1922666],
       [ 31285900,        25,     -8996,     -8995,     43204,  55146332],
       [ 31285900,         3,     -8996,     -8994,     43204,  55146332],
       [ 31283900,         8,     -8997,     -8994,     43204,  55146332],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0],
       [        0,         0,         0,         0,         0,         0]])

# quantities = agentTrades[:, 1]
# cut_idx = jnp.argmax(jnp.cumsum(quantities) > remainQuant)
# # Handle special case when all quantities are smaller than remainQuant
# cut_idx = jnp.where(cut_idx == 0, len(quantities), cut_idx)
# remaining_in_cut_idx = remainQuant - jnp.where(cut_idx > 0, jnp.cumsum(quantities)[cut_idx - 1], 0)



# quantities = agentTrades[:, 1]
# cumsum_quantities = jnp.cumsum(quantities)
# cut_idx = jnp.argmax(cumsum_quantities > remainQuant)

# # Handle special case when all quantities are smaller than remainQuant
# cut_idx = jnp.where(cut_idx == 0, len(quantities), cut_idx)

# remaining_in_cut_idx = remainQuant - jnp.where(cut_idx > 0, cumsum_quantities[cut_idx - 1], 0)

# new_second_col = jnp.where(jnp.arange(len(quantities)) == cut_idx, remaining_in_cut_idx, quantities)
# new_second_col = jnp.where(jnp.arange(len(quantities)) > cut_idx, 0, new_second_col)

# truncated_agentTrades = jnp.where(jnp.arange(len(quantities))[:, None] >= cut_idx, 0, agentTrades)
# truncated_agentTrades = truncated_agentTrades.at[:, 1].set(new_second_col)

# def truncate_agent_trades(agentTrades, remainQuant):
#     quantities = agentTrades[:, 1]
#     total_quant = jnp.sum(quantities)
    
#     # If remaining quantity is greater than total quantity, return original agentTrades
#     if remainQuant >= total_quant:
#         return agentTrades
    
#     cumsum_quantities = jnp.cumsum(quantities)

#     # If remaining quantity is smaller than the first quantity, return immediately.
#     if remainQuant <= quantities[0]:
#         new_agentTrades = jnp.zeros_like(agentTrades)
#         new_agentTrades = new_agentTrades.at[0, :].set(agentTrades[0])
#         new_agentTrades = new_agentTrades.at[0, 1].set(remainQuant)
#         return new_agentTrades
    
#     cut_idx = jnp.argmax(cumsum_quantities >= remainQuant)
#     remaining_in_cut_idx = remainQuant - cumsum_quantities[cut_idx - 1]
    
#     new_second_col = jnp.where(jnp.arange(len(quantities)) < cut_idx, quantities, 0)
#     new_second_col = jnp.where(jnp.arange(len(quantities)) == cut_idx, remaining_in_cut_idx, new_second_col)
    
#     truncated_agentTrades = agentTrades.at[:, 1].set(new_second_col)
    
#     zero_row = jnp.zeros_like(agentTrades[0])
#     truncated_agentTrades = jnp.where(jnp.arange(len(quantities))[:, None] > cut_idx, zero_row, truncated_agentTrades)
    
#     return truncated_agentTrades
remainQuant = 2
remainQuant = 25
remainQuant = 23
remainQuant = 230
remainQuant = 60
remainQuant = 61
remainQuant = 59
# remainQuant = 52

# def truncate_agent_trades(agentTrades, remainQuant):
    
#     quantities = agentTrades[:, 1]
#     if remainQuant >= jnp.sum(quantities):
#         return agentTrades
    
#     elif remainQuant <= quantities[0]:
#         new_agentTrades = jnp.zeros_like(agentTrades)
#         new_agentTrades = new_agentTrades.at[0, :].set(agentTrades[0])
#         new_agentTrades = new_agentTrades.at[0, 1].set(remainQuant)
#         return new_agentTrades

#     elif quantities[0] < remainQuant < jnp.sum(quantities):
#         cumsum_quantities = jnp.cumsum(quantities)
#         cut_idx = jnp.argmax(cumsum_quantities >= remainQuant)
#         remaining_in_cut_idx = remainQuant - cumsum_quantities[cut_idx - 1]
        
#         new_second_col = jnp.where(jnp.arange(len(quantities)) < cut_idx, quantities, 0)
#         new_second_col = jnp.where(jnp.arange(len(quantities)) == cut_idx, remaining_in_cut_idx, new_second_col)
        
#         truncated_agentTrades = agentTrades.at[:, 1].set(new_second_col)
#         zero_row = jnp.zeros_like(agentTrades[0])
#         truncated_agentTrades = jnp.where(jnp.arange(len(quantities))[:, None] > cut_idx, zero_row, truncated_agentTrades)
        
#         return truncated_agentTrades


remainQuant = 2
# remainQuant = 25
remainQuant = 23
remainQuant = 230
remainQuant = 60
# remainQuant = 61
# remainQuant = 59
remainQuant = 52

import jax.numpy as jnp

def truncate_agent_trades(agentTrades, remainQuant):
    quantities = agentTrades[:, 1]
    cumsum_quantities = jnp.cumsum(quantities)
    cut_idx = jnp.argmax(cumsum_quantities >= remainQuant)
    truncated_agentTrades = jnp.where(jnp.arange(len(quantities))[:, jnp.newaxis] > cut_idx, jnp.zeros_like(agentTrades[0]), agentTrades.at[:, 1].set(jnp.where(jnp.arange(len(quantities)) < cut_idx, quantities, jnp.where(jnp.arange(len(quantities)) == cut_idx, remainQuant - cumsum_quantities[cut_idx - 1], 0))))
    return jnp.where(remainQuant >= jnp.sum(quantities), agentTrades, jnp.where(remainQuant <= quantities[0], jnp.zeros_like(agentTrades).at[0, :].set(agentTrades[0]).at[0, 1].set(remainQuant), truncated_agentTrades))

def truncate_agent_trades(agentTrades, remainQuant):
    quantities = agentTrades[:, 1]
    cumsum_quantities = jnp.cumsum(quantities)
    cut_idx = jnp.argmax(cumsum_quantities >= remainQuant)
    
    truncated_agentTrades = jnp.where(
        jnp.arange(len(quantities))[:, jnp.newaxis] > cut_idx,
        jnp.zeros_like(agentTrades[0]),
        agentTrades.at[:, 1].set(
            jnp.where(
                jnp.arange(len(quantities)) < cut_idx,
                quantities,
                jnp.where(
                    jnp.arange(len(quantities)) == cut_idx,
                    remainQuant - cumsum_quantities[cut_idx - 1],
                    0
                )
            )
        )
    )
    
    return jnp.where(
        remainQuant >= jnp.sum(quantities),
        agentTrades,
        jnp.where(
            remainQuant <= quantities[0],
            jnp.zeros_like(agentTrades).at[0, :].set(agentTrades[0]).at[0, 1].set(remainQuant),
            truncated_agentTrades
        )
    )

# def truncate_agent_trades(agentTrades, remainQuant):
#     quantities = agentTrades[:, 1]
#     new_agentTrades = jnp.zeros_like(agentTrades)

#     new_agentTrades = new_agentTrades.at[0, :].set(agentTrades[0])
#     new_agentTrades = new_agentTrades.at[0, 1].set(remainQuant)

#     cumsum_quantities = jnp.cumsum(quantities)
#     cut_idx = jnp.argmax(cumsum_quantities >= remainQuant)
#     remaining_in_cut_idx = remainQuant - cumsum_quantities[cut_idx - 1]

#     new_second_col = jnp.where(jnp.arange(len(quantities)) < cut_idx, quantities, 0)
#     new_second_col = jnp.where(jnp.arange(len(quantities)) == cut_idx, remaining_in_cut_idx, new_second_col)

#     truncated_agentTrades = agentTrades.at[:, 1].set(new_second_col)
#     zero_row = jnp.zeros_like(agentTrades[0])
#     truncated_agentTrades = jnp.where(jnp.arange(len(quantities))[:, None] > cut_idx, zero_row, truncated_agentTrades)

#     return jnp.where(remainQuant >= jnp.sum(quantities), agentTrades, jnp.where(remainQuant <= quantities[0], new_agentTrades, truncated_agentTrades))

truncated_agentTrades = truncate_agent_trades(agentTrades, remainQuant)
truncated_agentTrades


agentTrades[:,1].sum()