from typing import Callable, Optional, Sequence

import distrax
import flax.linen as nn
import jax.numpy as jnp

from MCEP.networks import MLP
from MCEP.networks.constants import default_init


class SoftmaxPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    dropout_rate: Optional[float] = None
    apply_tanh: bool = True
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, training: bool = False
    ) -> distrax.Distribution:
        outputs = MLP(
            self.hidden_dims,
            activate_final=True,
            dropout_rate=self.dropout_rate,
            activations=self.activations,
        )(observations, training=training)

        means = nn.Dense(self.action_dim, kernel_init=default_init())(outputs)

        probs = nn.softmax(means)
        return probs