"""Implementations of algorithms for continuous control."""

import copy
import functools
from typing import Dict, Optional, Sequence, Tuple

import gym
import jax
import numpy as np
import optax
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState

from MCEP.agents.agent import Agent
from MCEP.agents.awac.actor_updater import update_actor
from MCEP.agents.awac.critic_updater import update_critic
from MCEP.networks.normal_tanh_policy import NormalTanhPolicy
from MCEP.networks.values import StateActionEnsemble
from MCEP.types import Params, PRNGKey
from MCEP.utils.target_update import soft_target_update


@functools.partial(jax.jit, static_argnames=("critic_reduction"))
def _update_jit(
    rng: PRNGKey,
    actor: TrainState,
    critic: TrainState,
    target_critic_params: Params,
    batch: FrozenDict,
    awac_lambda: float,
    discount: float,
    tau: float,
    critic_reduction: str,
) -> Tuple[PRNGKey, TrainState, TrainState, Params, TrainState, Dict[str, float]]:

    rng, key = jax.random.split(rng)
    target_critic = critic.replace(params=target_critic_params)
    new_critic, critic_info = update_critic(
        key,
        actor,
        critic,
        target_critic,
        batch,
        discount,
        critic_reduction=critic_reduction,
    )

    rng, key = jax.random.split(rng)
    new_actor, actor_info = update_actor(key, actor, new_critic, batch, awac_lambda)

    new_target_critic_params = soft_target_update(
        new_critic.params, target_critic_params, tau
    )

    return (
        rng,
        new_actor,
        new_critic,
        new_target_critic_params,
        {**critic_info, **actor_info},
    )


class AWACLearner(Agent):
    def __init__(
        self,
        seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        num_critics: int = 2,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256),
        awac_lambda: float = 1.0,
        discount: float = 0.99,
        tau: float = 0.005,
        critic_reduction: str = "min",
    ):
        """
        """

        action_dim = action_space.shape[-1]

        self.critic_reduction = critic_reduction

        self.awac_lambda = awac_lambda
        self.tau = tau
        self.discount = discount

        observations = observation_space.sample()
        actions = action_space.sample()

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)

        if np.all(action_space.low == -1) and np.all(action_space.high == 1):
            low = None
            high = None
        else:
            low = action_space.low
            high = action_space.high

        actor_def = NormalTanhPolicy(hidden_dims, action_dim, low=low, high=high)
        actor_params = actor_def.init(actor_key, observations)["params"]
        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=actor_lr),
        )

        critic_def = StateActionEnsemble(hidden_dims, num_qs=num_critics)
        critic_params = critic_def.init(critic_key, observations, actions)["params"]
        critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=optax.adam(learning_rate=critic_lr),
        )
        target_critic_params = copy.deepcopy(critic_params)

        self._actor = actor
        self._critic = critic
        self._target_critic_params = target_critic_params
        self._rng = rng

    def update(self, batch: FrozenDict) -> Dict[str, float]:
        (
            new_rng,
            new_actor,
            new_critic,
            new_target_critic_params,
            info,
        ) = _update_jit(
            self._rng,
            self._actor,
            self._critic,
            self._target_critic_params,
            batch,
            self.awac_lambda,
            self.discount,
            self.tau,
            self.critic_reduction,
        )

        self._rng = new_rng
        self._actor = new_actor
        self._critic = new_critic
        self._target_critic_params = new_target_critic_params

        return info