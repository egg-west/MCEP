from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState

from MCEP.types import Params, PRNGKey



def update_q(
    key: PRNGKey,
    target_actor: TrainState,
    critic: TrainState,
    target_critic: TrainState,
    batch: FrozenDict,
    discount: float,
    act_min: float,
    act_max: float,
    act_clip: float = 0.5,
    act_noise: float = 0.2,
    critic_reduction: str = "min"
) -> Tuple[TrainState, Dict[str, float]]:
    next_a = target_actor.apply_fn({"params": target_actor.params}, batch["next_observations"])
    noise = jnp.clip(jax.random.normal(key, next_a.shape) * act_noise,
                     -act_clip,
                     act_clip)
    next_a = jnp.clip(next_a + noise,
                      act_min,
                      act_max)

    next_qs = target_critic.apply_fn({"params": target_critic.params}, batch["next_observations"], next_a)
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
        return critic_loss, {"critic_loss": critic_loss, "q": qs.mean()}

    grads, info = jax.grad(critic_loss_fn, has_aux=True)(critic.params)
    new_critic = critic.apply_gradients(grads=grads)

    return new_critic, info
