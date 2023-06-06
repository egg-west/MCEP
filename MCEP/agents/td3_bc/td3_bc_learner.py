"""Implementations of algorithms for continuous control."""

import copy
import functools
from typing import Dict, Optional, Sequence, Tuple

import gym
import jax
import jax.numpy as jnp
import optax
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState

from MCEP.agents.agent import Agent
from MCEP.agents.td3_bc.actor_updater import update_actor
from MCEP.agents.td3_bc.critic_updater import update_q
from MCEP.networks import DeterministicPolicy
from MCEP.networks.values import StateActionEnsemble
from MCEP.types import Params, PRNGKey
from MCEP.utils.target_update import soft_target_update


@functools.partial(jax.jit, static_argnames=("critic_reduction","update_target_actor","behavior_cloning"))
def _update_jit(
    rng: PRNGKey,
    actor: TrainState,
    target_actor_params: Params,
    critic: TrainState,
    target_critic_params: Params,
    batch: TrainState,
    update_target_actor: bool,
    discount: float,
    act_min: float,
    act_max: float,
    act_clip: float,
    act_noise: float,
    tau: float,
    alpha: float,
    behavior_cloning: bool,
    bc_loss_weight: float,
    critic_reduction: str,
) -> Tuple[PRNGKey, TrainState, Params, TrainState, Params, Dict[str, float]]:

    key, rng = jax.random.split(rng)

    target_actor = actor.replace(params=target_actor_params)
    target_critic = critic.replace(params=target_critic_params)
    new_critic, critic_info = update_q(key,
                                       target_actor,
                                       critic,
                                       target_critic,
                                       batch,
                                       discount,
                                       act_min,
                                       act_max,
                                       act_clip,
                                       act_noise,
                                       critic_reduction)

    if update_target_actor:
        new_actor, actor_info = update_actor(key,
                                             actor,
                                             new_critic,
                                             batch,
                                             behavior_cloning,
                                             bc_loss_weight,
                                             alpha)

        new_target_critic_params = soft_target_update(
            new_critic.params, target_critic_params, tau
        )

        new_target_actor_params = soft_target_update(
            new_actor.params, target_actor_params, tau
        )
    else:
        new_actor, actor_info = actor, {}
        new_target_critic_params = target_critic_params
        new_target_actor_params = target_actor_params

    return (
        rng,
        new_actor,
        new_target_actor_params,
        new_critic,
        new_target_critic_params,
        {**critic_info, **actor_info},
    )


class TD3BCLearner(Agent):
    def __init__(
        self,
        seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        decay_steps: Optional[int] = None,
        hidden_dims: Sequence[int] = (256, 256),
        policy_delay: int = 2,
        act_noise: float = 0.2,
        act_clip: float = 0.5,
        discount: float = 0.99,
        tau: float = 0.005,
        alpha: float = 2.5,
        behavior_cloning: bool = False,
        bc_loss_weight: float = 1.0,
        critic_reduction: str = "min",
        apply_tanh: bool = True,
        dropout_rate: Optional[float] = None,
    ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1801.01290
        """
        observations = observation_space.sample()
        actions = action_space.sample()

        self.policy_delay = policy_delay
        self.act_noise = act_noise
        self.act_clip = act_clip
        self.act_max = action_space.high[0]
        self.act_min = action_space.low[0]
        self.discount = discount
        self.critic_reduction = critic_reduction
        self.behavior_cloning = behavior_cloning
        self.bc_loss_weight = bc_loss_weight
        self.tau=tau
        self.alpha=alpha

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key = jax.random.split(rng, 3)

        action_dim = actions.shape[-1]
        actor_def = DeterministicPolicy(
            hidden_dims, action_dim, self.act_max, dropout_rate=dropout_rate, apply_tanh=apply_tanh
        )

        if decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(actor_lr, decay_steps)

        actor_params = actor_def.init(actor_key, observations)["params"]
        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=actor_lr),
        )
        target_actor_params = copy.deepcopy(actor_params)

        critic_def = StateActionEnsemble(hidden_dims, num_qs=2)
        critic_params = critic_def.init(critic_key, observations, actions)["params"]
        critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=optax.adam(learning_rate=critic_lr),
        )
        target_critic_params = copy.deepcopy(critic_params)

        self._rng = rng
        self._actor = actor
        self._target_actor_params = target_actor_params
        self._critic = critic
        self._target_critic_params = target_critic_params

    def update(self, batch: FrozenDict, update_target_actor: bool) -> Dict[str, float]:
        (
            new_rng,
            new_actor,
            new_target_actor,
            new_critic,
            new_target_critic,
            info,
        ) = _update_jit(
            self._rng,
            self._actor,
            self._target_actor_params,
            self._critic,
            self._target_critic_params,
            batch,
            update_target_actor,
            self.discount,
            self.act_min,
            self.act_max,
            self.act_clip,
            self.act_noise,
            self.tau,
            self.alpha,
            self.behavior_cloning,
            self.bc_loss_weight,
            self.critic_reduction,
        )

        self._rng = new_rng
        self._actor = new_actor
        self._target_actor_params = new_target_actor
        self._critic = new_critic
        self._target_critic_params = new_target_critic

        return info
