from typing import Callable, Optional, Sequence

import distrax
import flax.linen as nn
import jax.numpy as jnp

from MCEP.networks import MLP
from MCEP.networks.constants import default_init

class UnitStdNormalPolicy(nn.Module):
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

        if self.apply_tanh:
            means = nn.tanh(means)

        return distrax.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.ones_like(means)
        )

class NormalPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    dropout_rate: Optional[float] = None
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2
    low: Optional[jnp.ndarray] = None
    high: Optional[jnp.ndarray] = None

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, training: bool = False
    ) -> distrax.Distribution:
        outputs = MLP(
            self.hidden_dims, activate_final=True, dropout_rate=self.dropout_rate
        )(observations, training=training)

        means = nn.Dense(self.action_dim, kernel_init=default_init())(outputs)

        log_stds = nn.Dense(self.action_dim, kernel_init=default_init())(outputs)

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        return distrax.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.exp(log_stds)
        )

class NormalLatentActionPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    dropout_rate: Optional[float] = None
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2
    low: Optional[jnp.ndarray] = None
    high: Optional[jnp.ndarray] = None

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, actions: jnp.ndarray, training: bool = False
    ) -> distrax.Distribution:
        inputs = {"states": observations, "actions": actions}
        outputs = MLP(
            self.hidden_dims, activate_final=True, dropout_rate=self.dropout_rate
        )(inputs, training=training)

        means = nn.Dense(self.action_dim, kernel_init=default_init())(outputs)

        log_stds = nn.Dense(self.action_dim, kernel_init=default_init())(outputs)

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        return distrax.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.exp(log_stds)
        )