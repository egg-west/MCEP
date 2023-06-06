from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState

from MCEP.types import Params, PRNGKey

def update_actor(
    key: PRNGKey,
    actor: TrainState,
    critic: TrainState,
    batch: FrozenDict,
    behavior_cloning: bool=False,
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
        if behavior_cloning:
            raw_bc = ((action - batch["actions"])**2)#.mean()
            bc_normalizer = jax.lax.stop_gradient(raw_bc.sum(axis=1).mean())
            bc_loss = raw_bc.mean()

            q_normalizer = jax.lax.stop_gradient(alpha / jnp.absolute(q).mean())
            actor_loss *= q_normalizer
            actor_loss += bc_loss / bc_normalizer
            log_dict["bc_loss"] = bc_loss

        return actor_loss, log_dict

    grads, info = jax.grad(actor_loss_fn, has_aux=True)(actor.params)
    new_actor = actor.apply_gradients(grads=grads)

    return new_actor, info