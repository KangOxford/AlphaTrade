import jax
import jax.numpy as jnp


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
        # print(a * max_sum / a_sum)
        a, remainders = jnp.divmod(a * max_sum, a_sum)
        rest = max_sum - jnp.sum(a)
        # indices of 'a' sorted by remainders in descending order (LTR priority tie-breaker)
        indices = (remainders.shape[0] - 1 - jnp.argsort(remainders[::-1]))[::-1]
        # ranks of remainders, highest starts with 0 
        ranks = jnp.argsort(indices)
        
        # print('remainders', remainders)
        # print('indices', indices)
        # print('ranks', ranks)
        
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
