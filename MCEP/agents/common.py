from functools import partial
from typing import Callable, Tuple

import distrax
import jax
import jax.numpy as jnp
import numpy as np

from MCEP.data.dataset import DatasetDict
from MCEP.types import Params, PRNGKey


@partial(jax.jit, static_argnames="actor_apply_fn")
def eval_log_prob_jit(
    actor_apply_fn: Callable[..., distrax.Distribution],
    actor_params: Params,
    batch: DatasetDict,
) -> float:
    dist = actor_apply_fn({"params": actor_params}, batch["observations"])
    log_probs = dist.log_prob(batch["actions"])
    return log_probs.mean()


@partial(jax.jit, static_argnames="actor_apply_fn")
def eval_actions_jit(
    actor_apply_fn: Callable[..., distrax.Distribution],
    actor_params: Params,
    observations: np.ndarray,
) -> jnp.ndarray:
    dist = actor_apply_fn({"params": actor_params}, observations)
    if isinstance(dist, distrax.Distribution):
        return dist.mode()
    else:
        return dist

@partial(jax.jit, static_argnames="actor_apply_fn")
def eval_MSEBCLoss_jit(
    actor_apply_fn: Callable[..., distrax.Distribution],
    actor_params: Params,
    observations: np.ndarray,
    actions: np.ndarray,
) -> jnp.ndarray:
    actor_action = actor_apply_fn({"params": actor_params}, observations)
    return ((actor_action - actions)**2).mean(axis=1)

@partial(jax.jit, static_argnames="actor_apply_fn")
def eval_LOGPBCLoss_jit(
    actor_apply_fn: Callable[..., distrax.Distribution],
    actor_params: Params,
    observations: np.ndarray,
    actions: np.ndarray,
) -> jnp.ndarray:
    pi_dist = actor_apply_fn({"params": actor_params}, observations)
    log_probs = pi_dist.log_prob(actions)
    return log_probs#((actor_action - actions)**2).mean(axis=1)


@partial(jax.jit, static_argnames=("actor_apply_fn", "actor_apply_fn2"))
def eval_KL_jit(
    actor_apply_fn: Callable[..., distrax.Distribution],
    actor_params: Params,
    actor_apply_fn2: Callable[..., distrax.Distribution],
    actor_params2: Params,
    observations: np.ndarray,
) -> jnp.ndarray:
    pi_dist = actor_apply_fn({"params": actor_params}, observations)
    pi_dist2 = actor_apply_fn2({"params": actor_params2}, observations)
    kl = pi_dist.kl_divergence(pi_dist2)
    rkl = pi_dist2.kl_divergence(pi_dist)
    #log_probs = pi_dist.log_prob(actions)
    return kl, rkl

@partial(jax.jit, static_argnames=("critic_apply_fn", "actor_apply_fn"))
def eval_q_jit(
    actor_apply_fn: Callable[..., distrax.Distribution],
    actor_params: Params,
    critic_apply_fn: Callable[..., distrax.Distribution],
    critic_params: Params,
    observations: np.ndarray,
    actions: np.ndarray,
) -> jnp.ndarray:
    dist = actor_apply_fn({"params": actor_params}, observations)
    if isinstance(dist, distrax.Distribution):
        predicted_actions = dist.mode()
    else:
        predicted_actions = dist
    qs = critic_apply_fn({"params": critic_params}, observations, predicted_actions)

    qs_data = critic_apply_fn({"params": critic_params}, observations, actions)
    return qs.mean(axis=0), qs_data.mean(axis=0)

@partial(jax.jit, static_argnames="actor_apply_fn")
def sample_actions_jit(
    rng: PRNGKey,
    actor_apply_fn: Callable[..., distrax.Distribution],
    actor_params: Params,
    observations: np.ndarray,
) -> Tuple[PRNGKey, jnp.ndarray]:
    dist = actor_apply_fn({"params": actor_params}, observations)
    rng, key = jax.random.split(rng)
    return rng, dist.sample(seed=key)