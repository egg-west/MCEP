from typing import Callable, Sequence

import flax.linen as nn
import jax.numpy as jnp

from MCEP.networks.mlps import MLP


class ActionValue(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(
        self, actions: jnp.ndarray, training: bool = False
    ) -> jnp.ndarray:

        inputs = {"actions": actions}
        critic = MLP((*self.hidden_dims, 1), activations=self.activations)(
            inputs, training=training
        )
        return jnp.squeeze(critic, -1)