from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from MCEP.data.dataset import DatasetDict
from MCEP.types import Params, PRNGKey


def update_critic(
    key: PRNGKey,
    actor: TrainState,
    critic: TrainState,
    target_critic: TrainState,
    batch: DatasetDict,
    discount: float,
    critic_reduction: str,
) -> Tuple[TrainState, Dict[str, float]]:
    dist = actor.apply_fn({"params": actor.params}, batch["next_observations"])
    next_actions, next_log_probs = dist.sample_and_log_prob(seed=key)
    next_qs = target_critic.apply_fn(
        {"params": target_critic.params}, batch["next_observations"], next_actions
    )
    if critic_reduction == "min":
        next_q = next_qs.min(axis=0)
    elif critic_reduction == "mean":
        next_q = next_qs.mean(axis=0)
    else:
        raise NotImplemented()

    target_q = batch["rewards"] + discount * batch["masks"] * next_q

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        qs = critic.apply_fn(
            {"params": critic_params}, batch["observations"], batch["actions"]
        )
        critic_loss = ((qs - target_q) ** 2).mean()
        return critic_loss, {
            "critic_loss": critic_loss,
            "q": qs.mean(),
            "target_actor_entropy": -next_log_probs.mean(),
        }

    grads, info = jax.grad(critic_loss_fn, has_aux=True)(critic.params)
    new_critic = critic.apply_gradients(grads=grads)

    return new_critic, info