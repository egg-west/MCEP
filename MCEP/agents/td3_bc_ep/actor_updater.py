from typing import Dict, Tuple, Any

import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState

from MCEP.types import Params, PRNGKey

def update_actor_bc_norm(
    key: PRNGKey,
    actor: TrainState,
    target_evaluation_actor: TrainState,
    bp_target_actor: TrainState,
    critic: TrainState,
    batch: FrozenDict,
    bc_normalizer_last_step: FrozenDict,
    bc_loss_weight: float=1.0,
    alpha: float = 2.5,
    discount: float = 0.99,
) -> Tuple[TrainState, Dict[str, float]]:

    tilde_action = bp_target_actor.apply_fn(
        {"params": bp_target_actor.params},
        batch["observations"],
        training=False,
        rngs={"dropout": key},
    )
    deviation = tilde_action - batch["actions"]
    bc_normalizer_this_step = jax.lax.stop_gradient((deviation**2).sum(axis=1).mean())
    bc_normalizer = 0.05 * bc_normalizer_this_step + 0.95 * bc_normalizer_last_step["bc_normalizer"]

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        action = actor.apply_fn(
            {"params": actor_params},
            batch["observations"],
            training=True,
            rngs={"dropout": key},
        )
        q = critic.apply_fn(
            {"params": critic.params}, batch["observations"], action
        )[0]

        actor_loss = -(q).mean()

        log_dict = {"actor_loss": actor_loss}

        bc_loss = ((action - batch["actions"])**2).mean()
        #raw_bc = (action - batch["actions"])
        #bc_loss = (raw_bc**2).mean()

        q_normalizer = jax.lax.stop_gradient(alpha / jnp.absolute(q).mean())
        actor_loss *= q_normalizer
        #actor_loss += bc_loss_weight * bc_loss
        actor_loss += bc_loss / bc_normalizer
        log_dict["bc_loss"] = bc_loss
        log_dict["bc_normalizer"] = bc_normalizer
        log_dict["unified_normalizer"] = alpha * bc_normalizer

        return actor_loss, log_dict

    grads, info = jax.grad(actor_loss_fn, has_aux=True)(actor.params)
    new_actor = actor.apply_gradients(grads=grads)
    #new_bc_normalizer = info["bc_normalizer"]
    return new_actor, info


def update_actor(
    key: PRNGKey,
    actor: TrainState,
    target_evaluation_actor: TrainState,
    bp_target_actor: TrainState,
    critic: TrainState,
    batch: FrozenDict,
    bc_loss_weight: float=1.0,
    alpha: float = 2.5,
    discount: float = 0.99,
) -> Tuple[TrainState, Dict[str, float]]:

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        action = actor.apply_fn(
            {"params": actor_params},
            batch["observations"],
            training=True,
            rngs={"dropout": key},
        )
        q = critic.apply_fn(
            {"params": critic.params}, batch["observations"], action
        )[0]

        actor_loss = -(q).mean()

        log_dict = {"actor_loss": actor_loss}

        bc_loss = ((action - batch["actions"])**2).mean()
        log_dict["bc_loss"] = bc_loss

        q_normalizer = jax.lax.stop_gradient(alpha / jnp.absolute(q).mean())
        actor_loss *= q_normalizer
        actor_loss += bc_loss_weight * bc_loss

        return actor_loss, log_dict

    grads, info = jax.grad(actor_loss_fn, has_aux=True)(actor.params)
    new_actor = actor.apply_gradients(grads=grads)

    return new_actor, info

"""
def update_actor(
    key: PRNGKey,
    actor: TrainState,
    critic: TrainState,
    batch: FrozenDict,
    bc_loss_weight: float=1.0,
    alpha: float = 2.5
) -> Tuple[TrainState, Dict[str, float]]:


    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        action = actor.apply_fn(
            {"params": actor_params},
            batch["observations"],
            training=True,
            rngs={"dropout": key},
        )
        q = critic.apply_fn(
            {"params": critic.params}, batch["observations"], action
        )[0]

        actor_loss = -(q).mean()

        log_dict = {"actor_loss": actor_loss}

        bc_loss = ((action - batch["actions"])**2).mean()
        q_normalizer = jax.lax.stop_gradient(alpha / jnp.absolute(q).mean())
        actor_loss *= q_normalizer
        actor_loss += bc_loss_weight * bc_loss
        log_dict["bc_loss"] = bc_loss

        return actor_loss, log_dict

    grads, info = jax.grad(actor_loss_fn, has_aux=True)(actor.params)
    new_actor = actor.apply_gradients(grads=grads)

    return new_actor, info
#"""