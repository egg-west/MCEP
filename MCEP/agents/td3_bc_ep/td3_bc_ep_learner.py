"""Implementations of algorithms for continuous control."""

import copy
import functools
from typing import Dict, Optional, Sequence, Tuple

import gym
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState
from MCEP.agents.common import eval_actions_jit, eval_MSEBCLoss_jit, eval_q_jit

from MCEP.agents.agent import Agent
from MCEP.agents.td3_bc_ep.actor_updater import update_actor, update_actor_bc_norm
from MCEP.agents.td3_bc_ep.bp_actor_updater import update_bp_actor
from MCEP.agents.td3_bc_ep.critic_updater import update_q
from MCEP.networks import DeterministicPolicy
from MCEP.networks.values import StateActionEnsemble
from MCEP.types import Params, PRNGKey
from MCEP.utils.target_update import soft_target_update
from flax.core import frozen_dict


@functools.partial(jax.jit, static_argnames=("critic_reduction", "update_target_actor", "use_bcNorm"))
def _update_jit(
    rng: PRNGKey,
    bp_actor: TrainState,
    bp_target_actor_params: Params,
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
    bp_alpha: float,
    tp_alpha: float,
    bp_bc_loss_weight: float,
    tp_bc_loss_weight: float,
    critic_reduction: str,
    bcNorm: TrainState,
    use_bcNorm: bool,
) -> Tuple[PRNGKey, TrainState, Params, TrainState, Params, Dict[str, float]]:

    key, rng = jax.random.split(rng)

    bp_target_actor = bp_actor.replace(params=bp_target_actor_params)
    evaluation_target_actor = actor.replace(params=target_actor_params)
    target_critic = critic.replace(params=target_critic_params)
    new_critic, critic_info = update_q(key,
                                       bp_target_actor,
                                       evaluation_target_actor,
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
        if use_bcNorm:
            new_actor, actor_info = update_actor_bc_norm(key,
                                                actor,
                                                evaluation_target_actor,
                                                bp_target_actor,
                                                new_critic,
                                                batch,
                                                bcNorm,
                                                tp_bc_loss_weight,
                                                tp_alpha)
        else:
            new_actor, actor_info = update_actor(key,
                                                actor,
                                                evaluation_target_actor,
                                                bp_target_actor,
                                                new_critic,
                                                batch,
                                                tp_bc_loss_weight,
                                                tp_alpha)
        new_bp_actor, bp_actor_info = update_bp_actor(key,
                                                   bp_actor,
                                                   new_critic,
                                                   batch,
                                                   bp_bc_loss_weight,
                                                   bp_alpha)

        new_target_critic_params = soft_target_update(
            new_critic.params, target_critic_params, tau
        )

        new_target_actor_params = soft_target_update(
            new_actor.params, target_actor_params, tau
        )

        new_bp_target_actor_params = soft_target_update(
            new_bp_actor.params, bp_target_actor_params, tau
        )
    else:
        new_actor, actor_info = actor, {}
        new_bp_actor, bp_actor_info = bp_actor, {}
        new_target_critic_params = target_critic_params
        new_target_actor_params = target_actor_params
        new_bp_target_actor_params = bp_target_actor_params
        actor_info["bc_normalizer"] = bcNorm["bc_normalizer"]

    return (
        rng,
        new_bp_actor,
        new_bp_target_actor_params,
        new_actor,
        new_target_actor_params,
        new_critic,
        new_target_critic_params,
        {**critic_info, **actor_info, **bp_actor_info},
    )


class TD3BCEPLearner(Agent):
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
        bp_alpha: float = 2.5,
        tp_alpha: float = 2.5,
        bp_bc_loss_weight: float = 1.0,     # behaviour policy bc loss
        tp_bc_loss_weight: float = 0.5,     # target policy bc loss
        critic_reduction: str = "min",
        apply_tanh: bool = True,
        dropout_rate: Optional[float] = None,
        initial_bcNorm: float = 1.0,
        use_bcNorm: bool = False,
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
        self.bp_bc_loss_weight = bp_bc_loss_weight
        self.tp_bc_loss_weight = tp_bc_loss_weight
        self.tau=tau
        self.bp_alpha=bp_alpha
        self.tp_alpha=tp_alpha
        self._bcNorm = initial_bcNorm
        self.use_bcNorm = use_bcNorm

        rng = jax.random.PRNGKey(seed)
        rng, bp_actor_key, actor_key, critic_key = jax.random.split(rng, 4)

        action_dim = actions.shape[-1]
        actor_def = DeterministicPolicy(
            hidden_dims, action_dim, self.act_max, dropout_rate=dropout_rate, apply_tanh=apply_tanh
        )

        if decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(actor_lr, decay_steps)

        bp_actor_params = actor_def.init(bp_actor_key, observations)["params"]
        bp_actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=bp_actor_params,
            tx=optax.adam(learning_rate=actor_lr),
        )
        bp_target_actor_params = copy.deepcopy(bp_actor_params)

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
        self._bp_actor = bp_actor
        self._bp_target_actor_params = bp_target_actor_params
        self._actor = actor
        self._target_actor_params = target_actor_params

        self._critic = critic
        self._target_critic_params = target_critic_params

    def update(self, batch: FrozenDict, update_target_actor: bool) -> Dict[str, float]:
        (
            new_rng,
            new_bp_actor,
            new_bp_target_actor,
            new_actor,
            new_target_actor,
            new_critic,
            new_target_critic,
            info,
        ) = _update_jit(
            self._rng,
            self._bp_actor,
            self._bp_target_actor_params,
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
            self.bp_alpha,
            self.tp_alpha,
            self.bp_bc_loss_weight,
            self.tp_bc_loss_weight,
            self.critic_reduction,
            frozen_dict.freeze({"bc_normalizer": self._bcNorm}),
            self.use_bcNorm,
        )

        self._rng = new_rng
        self._bp_actor = new_bp_actor
        self._bp_target_actor_params = new_bp_target_actor
        self._actor = new_actor
        self._target_actor_params = new_target_actor
        self._critic = new_critic
        self._target_critic_params = new_target_critic
        if self.use_bcNorm:
            self._bcNorm = float(info["bc_normalizer"])

        return info

    def eval_surrogate_actions(self, observations: np.ndarray) -> np.ndarray:
        actions = eval_actions_jit(
            self._bp_actor.apply_fn, self._bp_actor.params, observations
        )

        return np.asarray(actions)

    def eval_bcLoss_qValue(self, observations: np.ndarray, actions: np.ndarray) -> np.ndarray:
        bcLoss = eval_MSEBCLoss_jit(
            self._actor.apply_fn, self._actor.params, observations, actions
        )

        bcLoss_tilde = eval_MSEBCLoss_jit(
            self._bp_actor.apply_fn, self._bp_actor.params, observations, actions
        )

        qValue_tilde, qValue_data = eval_q_jit(
            self._bp_actor.apply_fn, self._bp_actor.params,
            self._critic.apply_fn, self._critic.params, observations, actions
        )

        qValue, _ = eval_q_jit(
            self._actor.apply_fn, self._actor.params,
            self._critic.apply_fn, self._critic.params, observations, actions
        )

        return np.asarray(bcLoss), np.asarray(bcLoss_tilde), np.asarray(qValue), np.asarray(qValue_tilde), np.asarray(qValue_data)