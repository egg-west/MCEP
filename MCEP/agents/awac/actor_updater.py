from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from MCEP.data.dataset import DatasetDict
from MCEP.types import Params, PRNGKey


def update_actor(
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
    #weight = jax.lax.stop_gradient(jnp.clip(jnp.exp(adv / awac_lambda), a_max=exp_adv_max))\
    weight = jnp.clip(jnp.exp(adv / awac_lambda), a_max=exp_adv_max)

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        pi_dist = actor.apply_fn({"params": actor_params}, batch["observations"])
        log_probs = pi_dist.log_prob(batch["actions"])
        actor_loss = (-log_probs * weight).mean()
        return actor_loss, {"actor_loss": actor_loss, "entropy": -log_probs.mean()}

    grads, info = jax.grad(actor_loss_fn, has_aux=True)(actor.params)
    new_actor = actor.apply_gradients(grads=grads)

    return new_actor, info