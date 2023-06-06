from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from flax.core.frozen_dict import FrozenDict

from MCEP.data.dataset import DatasetDict
from MCEP.types import Params, PRNGKey


def update_actor(
    key: PRNGKey,
    actor: TrainState,
    critic: TrainState,
    batch: DatasetDict,
    awac_lambda: float = 1.0,
    exp_adv_max: float = 1000.0,
) -> Tuple[TrainState, Dict[str, float]]:

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        pi_dist = actor.apply_fn({"params": actor_params}, batch["observations"])
        log_probs = pi_dist.log_prob(batch["actions"])

        actions, _ = pi_dist.sample_and_log_prob(seed=key)
        qs = critic.apply_fn({"params": critic.params}, batch["observations"], actions)
        q = qs.mean(axis = 0)
        q = jnp.clip(q, a_max=exp_adv_max)
        #q_normalizer = jax.lax.stop_gradient(1.0 / jnp.absolute(q).mean())

        #actor_loss = (-(q / q_normalizer) - awac_lambda * log_probs).mean()
        actor_loss = (-q - awac_lambda * log_probs).mean()
        return actor_loss, {"actor_loss": actor_loss, "entropy": -log_probs.mean()}

    grads, info = jax.grad(actor_loss_fn, has_aux=True)(actor.params)
    new_actor = actor.apply_gradients(grads=grads)

    return new_actor, info


def update_actor_bc_norm(
    key: PRNGKey,
    actor: TrainState,
    actor_tilde: TrainState,
    critic: TrainState,
    batch: DatasetDict,
    bc_normalizer_last_step: FrozenDict,
    awac_lambda: float = 1.0,
    exp_adv_max: float = 10000.0,
) -> Tuple[TrainState, Dict[str, float]]:
    """normalize by 1.1^n"""
    pi_dist_tilde  = actor_tilde.apply_fn(
        {"params": actor_tilde.params},
        batch["observations"]
    )
    likelihood_tilde = pi_dist_tilde.log_prob(batch["actions"])
    bc_normalizer_this_step = 1.0 / (1.1**(-likelihood_tilde)).mean()

    bc_normalizer = 0.01 * bc_normalizer_this_step + 0.99 * bc_normalizer_last_step["bc_normalizer"]

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        pi_dist = actor.apply_fn({"params": actor_params}, batch["observations"])
        log_probs = pi_dist.log_prob(batch["actions"])

        actions, _ = pi_dist.sample_and_log_prob(seed=key)
        qs = critic.apply_fn({"params": critic.params}, batch["observations"], actions)
        q = qs.mean(axis = 0)

        actor_loss = (-q - bc_normalizer * log_probs).mean()
        return actor_loss, {"actor_loss": actor_loss, "entropy": -log_probs.mean()}

    grads, info = jax.grad(actor_loss_fn, has_aux=True)(actor.params)
    new_actor = actor.apply_gradients(grads=grads)

    info["bc_normalizer"] = bc_normalizer
    return new_actor, info

def update_actor_tilde(
    key: PRNGKey,
    actor: TrainState,
    critic: TrainState,
    batch: DatasetDict,
    awac_lambda: float = 1.0,
    exp_adv_max: float = 100.0,
) -> Tuple[TrainState, Dict[str, float]]:
    dist = actor.apply_fn({"params": actor.params}, batch["observations"])
    actions, _ = dist.sample_and_log_prob(seed=key)
    vs = critic.apply_fn({"params": critic.params}, batch["observations"], actions)
    v = vs.min(axis = 0)
    qs = critic.apply_fn({"params": critic.params}, batch["observations"], batch["actions"])
    q = qs.min(axis = 0)
    adv = q - v
    original_weight = jnp.exp(adv / awac_lambda)
    weight = jnp.clip(original_weight, a_max=exp_adv_max)

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        pi_dist = actor.apply_fn({"params": actor_params}, batch["observations"])
        log_probs = pi_dist.log_prob(batch["actions"])
        actor_loss = (-log_probs * weight).mean()
        return actor_loss, {"actor_tilde_loss": actor_loss, "actor_tilde_entropy": -log_probs.mean()}

    grads, info = jax.grad(actor_loss_fn, has_aux=True)(actor.params)
    new_actor = actor.apply_gradients(grads=grads)

    #info['original_weight_mean_tilde'] = original_weight.mean()
    #info['original_weight_max_tilde'] = original_weight.max()

    return new_actor, info