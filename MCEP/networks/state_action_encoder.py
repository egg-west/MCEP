from typing import Callable, Sequence

import flax.linen as nn
import jax.numpy as jnp

from MCEP.networks.mlps import MLP


class StateActionEncoder(nn.Module):
    hidden_dims: Sequence[int]
    latent_dim: int
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, actions: jnp.ndarray, training: bool = False
    ) -> jnp.ndarray:
        inputs = {"states": observations, "actions": actions}
        critic = MLP((*self.hidden_dims, self.latent_dim), activations=self.activations)(
            inputs, training=training
        )
        return critic