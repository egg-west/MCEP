from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState

from MCEP.types import Params, PRNGKey

def update_bp_actor(
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

        log_dict = {"bp_actor_loss": actor_loss}

        bc_loss = ((action - batch["actions"])**2).mean()
        q_normalizer = jax.lax.stop_gradient(alpha / jnp.absolute(q).mean())
        actor_loss *= q_normalizer
        actor_loss += bc_loss_weight * bc_loss
        log_dict["bp_bc_loss"] = bc_loss

        return actor_loss, log_dict

    grads, info = jax.grad(actor_loss_fn, has_aux=True)(actor.params)
    new_actor = actor.apply_gradients(grads=grads)

    return new_actor, info