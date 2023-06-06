from typing import Callable, Sequence

import flax.linen as nn
import jax.numpy as jnp

from MCEP.networks.values.action_value import ActionValue


class ActionValueEnsemble(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    num_qs: int = 5

    @nn.compact
    def __call__(self, actions, training: bool = False):

        VmapCritic = nn.vmap(
            ActionValue,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num_qs,
        )
        qs = VmapCritic(self.hidden_dims, activations=self.activations)(
            actions, training
        )
        return qs