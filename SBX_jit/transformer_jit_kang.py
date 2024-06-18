import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import flax.linen as nn
import jax.numpy as jnp
import jax

import numpy as np
from sbx.ppo.policies import PPOPolicy
from stable_baselines3.common.type_aliases import Schedule


class AttentionBlock(nn.Module):
    d_model: int = 128
    nhead: int = 2
    linear_hidden_dim: int = 2048

    @nn.compact
    def __call__(self, src, mask):
        self_attn = nn.SelfAttention(
            num_heads=self.nhead,
            qkv_features=self.d_model,
        )
        norm1 = nn.LayerNorm()
        norm2 = nn.LayerNorm()
        linear1 = nn.Dense(self.linear_hidden_dim)
        linear2 = nn.Dense(self.d_model)

        src2 = norm1(src)
        attn_output = self_attn(src2, mask=mask, deterministic=True)
        src = src + attn_output

        src2 = norm2(src)
        src2 = linear2(nn.relu(linear1(src2)))
        src = src + src2

        return src


class TransformerCritic(nn.Module):
    num_layers: int = 5
    d_model: int = 128
    nhead: int = 2
    linear_hidden_dim: int = 2048

    def setup(self):
        self.embedding_layer = nn.Dense(self.d_model)
        self.attention_layers = [AttentionBlock(d_model=self.d_model, nhead=self.nhead, linear_hidden_dim=self.linear_hidden_dim) for _ in range(self.num_layers)]
        self.decoder = nn.Dense(1)

    def __call__(self, inputs, mask):
        return self.forward(inputs, mask)

    def forward(self, inputs, mask):
        embedding = self.embedding_layer(inputs)
        for layer in self.attention_layers:
            embedding = layer(embedding, mask)
        output = self.decoder(embedding)
        return output


from flax.linen.initializers import constant
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

class TransformerActor(nn.Module):
    out_dim: int
    num_layers: int = 5
    d_model: int = 128
    nhead: int = 2
    linear_hidden_dim: int = 2048

    def setup(self):
        self.embedding_layer = nn.Dense(self.d_model)
        self.attention_layers = [AttentionBlock(d_model=self.d_model, nhead=self.nhead, linear_hidden_dim=self.linear_hidden_dim) for _ in range(self.num_layers)]
        self.decoder = nn.Dense(self.out_dim)
        self.log_std = self.param("log_std", constant(0.9), (self.out_dim))

    def __call__(self, inputs, mask):
        action_logits = self.forward(inputs, mask)
        dist = tfd.MultivariateNormalDiag(loc=action_logits, scale_diag=jnp.exp(self.log_std))
        return dist

    def forward(self, inputs, mask):
        embedding = self.embedding_layer(inputs)
        for layer in self.attention_layers:
            embedding = layer(embedding, mask)
        output = self.decoder(embedding)
        return output


from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import optax
import gym
from flax.training.train_state import TrainState

class TransformerPolicy(PPOPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        ortho_init: bool = False,
        log_std_init: float = 0.0,
        activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.tanh,
        use_sde: bool = False,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class=None,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Callable[..., optax.GradientTransformation] = optax.adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            ortho_init,
            log_std_init,
            activation_fn,
            use_sde,
            use_expln,
            clip_mean,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            share_features_extractor,
        )

    def build(self, key, lr_schedule, max_grad_norm):

        key, actor_key, vf_key = jax.random.split(key, 3)
        key, self.key = jax.random.split(key, 2)
        self.reset_noise()

        obs = jnp.array([self.observation_space.sample()]).reshape(1, 12, 52)
        mask = jnp.ones([1, 2, 12, 12])

        self.actor = TransformerActor(out_dim=2)
        self.actor.reset_noise = self.reset_noise
        self.actor_state = TrainState.create(
            apply_fn=self.actor.apply,
            params=self.actor.init(actor_key, obs, mask),
            tx=optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                self.optimizer_class(
                    learning_rate=lr_schedule(1),  # type: ignore[call-arg]
                    **self.optimizer_kwargs,  # , eps=1e-5
                ),
            ),
        )

        self.vf = TransformerCritic()
        self.vf_state = TrainState.create(
            apply_fn=self.vf.apply,
            params=self.vf.init({"params": vf_key}, obs, mask),
            tx=optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                self.optimizer_class(
                    learning_rate=lr_schedule(1),  # type: ignore[call-arg]
                    **self.optimizer_kwargs,  # , eps=1e-5
                ),
            ),
        )

        self.actor.apply = jax.jit(self.actor.apply)  # type: ignore[method-assign]
        self.vf.apply = jax.jit(self.vf.apply)  # type: ignore[method-assign]

        return key

    def predict_all(self, observation: jnp.ndarray, mask: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
        return self._predict_all(self.actor_state, self.vf_state, observation, mask, key)

    @staticmethod
    @jax.jit
    def _predict_all(actor_state, vf_state, observation, mask, key):
        dist = actor_state.apply_fn(actor_state.params, observation, mask)
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)
        values = vf_state.apply_fn(vf_state.params, observation, mask).mean(axis=1).flatten()
        return actions, log_probs, values

# Example usage
if __name__ == "__main__":
    import time
    import optax
    from flax.training.train_state import TrainState

    d_model = 128
    nhead = 2
    num_layers = 5
    batch_size = 32
    seq_len = 12
    feature_dim = 52
    out_dim = 2

    # Initialize the Transformer models
    actor_model = TransformerActor(out_dim=out_dim, num_layers=num_layers, d_model=d_model, nhead=nhead, linear_hidden_dim=2048)
    critic_model = TransformerCritic(num_layers=num_layers, d_model=d_model, nhead=nhead, linear_hidden_dim=2048)

    # Generate some random input data
    src = jnp.ones((batch_size, seq_len, feature_dim))
    mask = jnp.ones((batch_size, nhead, seq_len, seq_len))

    # Initialize the model parameters
    actor_variables = actor_model.init(jax.random.PRNGKey(0), src, mask)
    critic_variables = critic_model.init(jax.random.PRNGKey(0), src, mask)

    # JIT compile the apply functions
    actor_apply = jax.jit(actor_model.apply)
    critic_apply = jax.jit(critic_model.apply)

    # Warm up
    actor_apply(actor_variables, src, mask)
    critic_apply(critic_variables, src, mask)

    # Measure execution time for the actor model (inference)
    start_time = time.time()
    for _ in range(100):
        actor_apply(actor_variables, src, mask)
    actor_time = time.time() - start_time
    print(f"Execution time for actor model (inference): {actor_time:.4f} seconds")

    # Measure execution time for the critic model (inference)
    start_time = time.time()
    for _ in range(100):
        critic_apply(critic_variables, src, mask)
    critic_time = time.time() - start_time
    print(f"Execution time for critic model (inference): {critic_time:.4f} seconds")

    def actor_loss_fn(params, src, mask):
        dist = actor_model.apply({'params': params}, src, mask)
        actions = dist.sample(seed=jax.random.PRNGKey(1))
        log_probs = dist.log_prob(actions)
        loss = -jnp.mean(log_probs)  # Simplified loss for testing
        return loss


    def critic_loss_fn(params, src, mask):
        values = critic_model.apply({'params': params}, src, mask).flatten()
        target_values = jnp.ones_like(values)  # Simplified target for testing
        loss = jnp.mean((values - target_values) ** 2)
        return loss

    # Initialize the optimizers and training states
    actor_optimizer = optax.adam(learning_rate=1e-3)
    critic_optimizer = optax.adam(learning_rate=1e-3)

    actor_train_state = TrainState.create(apply_fn=actor_model.apply, params=actor_variables['params'], tx=actor_optimizer)
    critic_train_state = TrainState.create(apply_fn=critic_model.apply, params=critic_variables['params'], tx=critic_optimizer)

    # Define the training step functions
    @jax.jit
    def train_actor_step(state, src, mask):
        loss, grads = jax.value_and_grad(actor_loss_fn)(state.params, src, mask)
        state = state.apply_gradients(grads=grads)
        return state, loss

    @jax.jit
    def train_critic_step(state, src, mask):
        loss, grads = jax.value_and_grad(critic_loss_fn)(state.params, src, mask)
        state = state.apply_gradients(grads=grads)
        return state, loss

    # Measure training time for the actor model
    start_time = time.time()
    for _ in range(100):
        actor_train_state, actor_loss = train_actor_step(actor_train_state, src, mask)
    actor_train_time = time.time() - start_time
    print(f"Execution time for actor model (training): {actor_train_time:.4f} seconds")

    # Measure training time for the critic model
    start_time = time.time()
    for _ in range(100):
        critic_train_state, critic_loss = train_critic_step(critic_train_state, src, mask)
    critic_train_time = time.time() - start_time
    print(f"Execution time for critic model (training): {critic_train_time:.4f} seconds")
