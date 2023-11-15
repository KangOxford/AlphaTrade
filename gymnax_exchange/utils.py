import jax
import jax.numpy as jnp


def argsort_rev(arr):
    """ 'arr' sorted in descending order (LTR priority tie-breaker) """
    return (arr.shape[0] - 1 - jnp.argsort(arr[::-1]))[::-1]

def rank_rev(arr):
    """ Rank array in descending order, with ties having left-to-right priority. """
    return jnp.argsort(argsort_rev(arr))

@jax.jit
def clip_by_sum_int(a: jax.Array, max_sum: int) -> jax.Array:
    """ Clip a vector so that its sum is at most max_sum as an integer,
        while preserving the relative proportions of the elements.
        Ties have left-to-right priority.

        ex: clip_by_sum_int(jnp.array([3, 2, 3, 1]), 8)) -->  [3 2 2 1]

    Args:
        a: The vector to clip.
        max_sum: The maximum sum of the vector.

    Returns:
        The clipped vector.
    """
    def clip(a: jax.Array, a_sum: int) -> jax.Array:
        a, remainders = jnp.divmod(a * max_sum, a_sum)
        rest = max_sum - jnp.sum(a)
        ranks = rank_rev(remainders)
        
        # add 1 to first 'rest' elements of original 'a' with highest remainder
        a = jnp.where(
            ranks < rest,
            a + 1,
            a,
        )
        return a

    a_sum = jnp.sum(a)
    return jax.lax.cond(
        a_sum > max_sum,
        lambda: clip(a, a_sum),
        lambda: a,
    )

from functools import partial
@partial(jax.vmap, in_axes=(0, None))
def p_in_cnl(p, prices_cnl):
    return jnp.where((prices_cnl == p) & (p != 0), True, False)
def matching_masks(prices_a, prices_cnl):
    res = p_in_cnl(prices_a, prices_cnl)
    return jnp.any(res, axis=1), jnp.any(res, axis=0)

#main
if __name__ == '__main__':
    print(
        matching_masks(
            jnp.array([1, 2, 100, 0]),
            jnp.array([100, 4, 4, 4])
        )
    )