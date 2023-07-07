import numpy as np
import timeit
mat = np.ones(600).reshape(-1,6)

a = mat
b = mat.reshape(-1,)
c = mat.tolist()


%timeit a[:,3]
%timeit b[3::5]
%timeit c[3::5]



# --------------------------------------------------------
num_envs = 1000*1000
mat = np.ones(600*num_envs).reshape(-1,6) 

a = mat
b = mat.reshape(-1,)
c = mat.tolist()


%timeit a[:,3]
%timeit b[3::5]
%timeit c[3::5]


In [4]: %timeit a[:,3]
207 ns ± 3.04 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)



In [6]: %timeit c[3::5]
395 ms ± 2.93 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

# time_a = timeit.timeit(lambda: a[:, 3], number=1)
# time_b = timeit.timeit(lambda: b[3::5], number=1)
# time_c = timeit.timeit(lambda: c[3::5], number=1)

# print("Time taken for 'a[:, 3]':", time_a)
# print("Time taken for 'b[3::5]':", time_b)
# print("Time taken for 'c[3::5]':", time_c)


# time_a = timeit.timeit(lambda: a[:, 3], number=10000)
# time_b = timeit.timeit(lambda: b[3::5], number=10000)
# time_c = timeit.timeit(lambda: c[3::5], number=10000)

# print("Time taken for 'a[:, 3]':", time_a)
# print("Time taken for 'b[3::5]':", time_b)
# print("Time taken for 'c[3::5]':", time_c)