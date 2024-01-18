import jax
import jax.numpy as jnp
import jax.tree_util as jtu

"""hamilton_apportionment_permuted_jax: A utility function using JAX, 
                                     implementing a Hamilton apportionment 
                                     method with randomized seat allocation."""

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


def tree_stack(trees):
    return jtu.tree_map(lambda *v: jnp.stack(v), *trees)

def tree_unstack(tree):
    leaves, treedef = jtu.tree_flatten(tree)
    return [treedef.unflatten(leaf) for leaf in zip(*leaves, strict=True)]

def array_index(array,index):
    return array[index]

@jax.jit
def index_tree(tree,index):
    array_index = lambda array,index : array[index]
    indeces=[index]*len(jtu.tree_flatten(tree)[0])
    tree_indeces=jtu.tree_unflatten(jtu.tree_flatten(tree)[1],indeces)
    return jtu.tree_map(array_index,tree,tree_indeces)

def hamilton_apportionment_permuted_jax(votes, seats, key):
    """
    Compute the Hamilton apportionment method with permutation using JAX.

    Args:
        votes (jax.Array): Array of votes for each party/entity.
        seats (int): Total number of seats to be apportioned.
        key (chex.PRNGKey): JAX key for random number generation.

    Returns:
        jax.Array: Array of allocated seats to each party/entity.
    """
    std_divisor = jnp.sum(votes) / seats # Calculate the standard divisor.
    # Initial allocation of seats based on the standard divisor and compute remainders.
    init_seats, remainders = jnp.divmod(votes, std_divisor)
    # Compute the number of remaining seats to be allocated.
    remaining_seats = jnp.array(seats - init_seats.sum(), dtype=jnp.int32) 
    # Define the scanning function for iterative seat allocation.
    def allocate_remaining_seats(carry,x): # only iterate 4 times, as remaining_seats in {0,1,2,3}
        key,init_seats,remainders = carry
        key, subkey = jax.random.split(key)
        # Create a probability distribution based on the maximum remainder.
        distribution = (remainders == remainders.max())/(remainders == remainders.max()).sum()
        # Randomly choose a party/entity to allocate a seat based on the distribution.
        chosen_index = jax.random.choice(subkey, remainders.size, p=distribution)
        # Update the initial seats and remainders for the chosen party/entity.
        updated_init_seats = init_seats.at[chosen_index].add(jnp.where(x < remaining_seats, 1, 0))
        updated_remainders = remainders.at[chosen_index].set(0)
        return (key, updated_init_seats, updated_remainders), x
        # Iterate over parties/entities to allocate the remaining seats.
    (key, init_seats, remainders), _ = jax.lax.scan(
                                                    allocate_remaining_seats,
                                                    (key, init_seats, remainders), 
                                                    xs=jnp.arange(votes.shape[0])
                                                    )
    return init_seats