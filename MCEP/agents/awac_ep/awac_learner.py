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
from MCEP.agents.awac_ep.actor_updater import update_actor, update_actor_tilde, update_actor_bc_norm
from MCEP.agents.awac_ep.critic_updater import update_critic
from MCEP.networks.normal_tanh_policy import NormalTanhPolicy
from MCEP.networks.values import StateActionEnsemble
from MCEP.networks import UnitStdNormalPolicy
from MCEP.types import Params, PRNGKey
from MCEP.utils.target_update import soft_target_update
from MCEP.agents.common import eval_actions_jit, eval_LOGPBCLoss_jit, eval_q_jit, eval_KL_jit
from flax.core import frozen_dict


@functools.partial(jax.jit, static_argnames=("critic_reduction", "evaluation_labmda", "tilde_lambda", "use_bc_norm", "exp_adv_max", "q_max"))
def _update_jit(
    rng: PRNGKey,
    actor: TrainState,
    actor_tilde: TrainState,
    critic: TrainState,
    target_critic_params: Params,
    batch: FrozenDict,
    bcNorm: TrainState,
    discount: float,
    tau: float,
    critic_reduction: str,
    evaluation_labmda: float=0.5,
    tilde_lambda: float=1.0,
    use_bc_norm: bool=False,
    exp_adv_max: float = 1000.0,
    q_max: float = 1000.0,
) -> Tuple[PRNGKey, TrainState, TrainState, Params, TrainState, Dict[str, float]]:

    rng, key = jax.random.split(rng)
    target_critic = critic.replace(params=target_critic_params)
    new_critic, critic_info = update_critic(
        key,
        actor_tilde,
        actor,
        critic,
        target_critic,
        batch,
        discount,
        critic_reduction=critic_reduction,
    )

    rng, key = jax.random.split(rng)

    new_actor_tilde, actor_tilde_info = update_actor_tilde(key, actor_tilde, new_critic, batch, awac_lambda=tilde_lambda, exp_adv_max=exp_adv_max)
    if use_bc_norm:
        new_actor, actor_info = update_actor_bc_norm(key, actor, actor_tilde, new_critic, batch, bcNorm, awac_lambda=evaluation_labmda)
    else:
        new_actor, actor_info = update_actor(key, actor, new_critic, batch, awac_lambda=evaluation_labmda, exp_adv_max=q_max)

    new_target_critic_params = soft_target_update(
        new_critic.params, target_critic_params, tau
    )

    return (
        rng,
        new_actor,
        new_actor_tilde,
        new_critic,
        new_target_critic_params,
        {**critic_info, **actor_info, **actor_tilde_info},
    )


class AWACEPLearner(Agent):
    def __init__(
        self,
        seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        num_critics: int = 2,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        critic_reduction: str = "min",
        evaluation_lambda: float = 0.5,
        tilde_lambda: float = 1.0,
        use_bc_norm: bool = False,
        initial_bcNorm: float = 1.0,
        exp_adv_max: float = 1000.0,
        q_max: float = 10000.0,
    ):
        """
        """

        action_dim = action_space.shape[-1]

        self.critic_reduction = critic_reduction

        self.tau = tau
        self.discount = discount
        self.evaluation_lambda = evaluation_lambda
        self.tilde_lambda = tilde_lambda
        self.use_bc_norm = use_bc_norm
        self._bcNorm = initial_bcNorm
        self.exp_adv_max = exp_adv_max
        self.q_max = q_max

        observations = observation_space.sample()
        actions = action_space.sample()

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, actor_tilde_key, critic_key, temp_key = jax.random.split(rng, 5)

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

        actor_tilde_def = NormalTanhPolicy(hidden_dims, action_dim, low=low, high=high)
        actor_tilde_params = actor_def.init(actor_key, observations)["params"]
        actor_tilde = TrainState.create(
            apply_fn=actor_tilde_def.apply,
            params=actor_tilde_params,
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
        self._actor_tilde = actor_tilde
        self._critic = critic
        self._target_critic_params = target_critic_params
        self._rng = rng

    def update(self, batch: FrozenDict) -> Dict[str, float]:
        (
            new_rng,
            new_actor,
            new_actor_tilde,
            new_critic,
            new_target_critic_params,
            info,
        ) = _update_jit(
            self._rng,
            self._actor,
            self._actor_tilde,
            self._critic,
            self._target_critic_params,
            batch,
            frozen_dict.freeze({"bc_normalizer": self._bcNorm}),
            self.discount,
            self.tau,
            self.critic_reduction,
            self.evaluation_lambda,
            self.tilde_lambda,
            self.use_bc_norm,
            self.exp_adv_max,
            self.q_max,
        )

        self._rng = new_rng
        self._actor = new_actor
        self._actor_tilde = new_actor_tilde
        self._critic = new_critic
        self._target_critic_params = new_target_critic_params
        if self.use_bc_norm:
            self._bcNorm = float(info["bc_normalizer"])

        return info

    def eval_surrogate_actions(self, observations: np.ndarray) -> np.ndarray:
        actions = eval_actions_jit(
            self._actor_tilde.apply_fn, self._actor_tilde.params, observations
        )

        return np.asarray(actions)

    def eval_bcLoss_qValue(self, observations: np.ndarray, actions: np.ndarray) -> np.ndarray:
        #bcLoss = eval_LOGPBCLoss_jit(
        #    self._actor.apply_fn, self._actor.params, observations, actions
        #)

        #bcLoss_tilde = eval_LOGPBCLoss_jit(
        #    self._actor_tilde.apply_fn, self._actor_tilde.params, observations, actions
        #)

        qValue_tilde, qValue_data = eval_q_jit(
            self._actor_tilde.apply_fn, self._actor_tilde.params,
            self._critic.apply_fn, self._critic.params, observations, actions
        )

        qValue, _ = eval_q_jit(
            self._actor.apply_fn, self._actor.params,
            self._critic.apply_fn, self._critic.params, observations, actions
        )

        kl, rkl = eval_KL_jit(
            self._actor.apply_fn, self._actor.params, self._actor_tilde.apply_fn, self._actor_tilde.params, observations
        )

        #return np.asarray(bcLoss), np.asarray(bcLoss_tilde), np.asarray(qValue), np.asarray(qValue_tilde), np.asarray(qValue_data)
        return np.asarray(kl), np.asarray(rkl), np.asarray(qValue), np.asarray(qValue_tilde), np.asarray(qValue_data)