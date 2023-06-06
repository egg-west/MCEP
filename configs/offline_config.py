import ml_collections
from ml_collections.config_dict import config_dict


def get_bc_config():
    config = ml_collections.ConfigDict()

    config.actor_lr = 1e-3
    config.hidden_dims = (256, 256)
    config.cosine_decay = True
    config.dropout_rate = 0.1
    config.weight_decay = config_dict.placeholder(float)
    config.apply_tanh = True

    config.distr = "unitstd_normal"
    # unitstd_normal | tanh_normal | ar_mog

    return config

def get_iql_mujoco_config():
    config = ml_collections.ConfigDict()

    config.actor_lr = 3e-4
    config.value_lr = 3e-4
    config.critic_lr = 3e-4

    config.hidden_dims = (256, 256)

    config.discount = 0.99

    config.expectile = 0.7  # The actual tau for expectiles.
    config.A_scaling = 3.0
    config.dropout_rate = config_dict.placeholder(float)
    config.cosine_decay = True

    config.tau = 0.005  # For soft target updates.

    config.critic_reduction = "min"

    return config

def get_td3bc_mujoco_config():
    config = ml_collections.ConfigDict()

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4

    config.hidden_dims = (256, 256)

    config.discount = 0.99

    config.behavior_cloning = True
    config.bc_loss_weight = 1.0
    config.alpha = 2.5

    config.policy_delay = 2
    config.act_noise = 0.2
    config.act_clip = 0.5
    config.tau = 0.005

    config.critic_reduction = "min"

    return config

def get_td3bcep_mujoco_config():
    config = ml_collections.ConfigDict()

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4

    config.hidden_dims = (256, 256)

    config.discount = 0.99

    config.bp_bc_loss_weight = 1.0
    config.tp_bc_loss_weight = 1.0
    config.bp_alpha = 2.5#2.5
    config.tp_alpha = 10.0#10.0

    config.policy_delay = 2
    config.act_noise = 0.2
    config.act_clip = 0.5
    config.tau = 0.005

    config.critic_reduction = "min"

    config.use_bcNorm = False

    return config

def get_awac_mujoco_config():
    config = ml_collections.ConfigDict()

    config.num_critics = 2

    config.actor_lr = 3e-5
    config.critic_lr = 3e-4

    config.hidden_dims = (256, 256)

    config.discount = 0.99
    config.tau = 0.005
    config.awac_lambda = 0.1

    config.critic_reduction = "min"

    return config

def get_awacep_mujoco_config():
    config = ml_collections.ConfigDict()

    config.num_critics = 2

    config.actor_lr = 3e-5
    config.critic_lr = 3e-4

    config.hidden_dims = (256, 256)

    config.discount = 0.99
    config.tau = 0.005

    config.critic_reduction = "min"

    config.evaluation_lambda = 0.6
    config.tilde_lambda = 1.0

    config.use_bc_norm = False

    return config


def get_config(config_string):
    possible_structures = {
        "bc": ml_collections.ConfigDict(
            {"model_constructor": "BCLearner", "model_config": get_bc_config()}
        ),
        "iql_mujoco": ml_collections.ConfigDict(
            {"model_constructor": "IQLLearner", "model_config": get_iql_mujoco_config()}
        ),
        "td3bc_mujoco": ml_collections.ConfigDict(
            {"model_constructor": "TD3BCLearner", "model_config": get_td3bc_mujoco_config()}
        ),
        "td3bcep_mujoco": ml_collections.ConfigDict(
            {"model_constructor": "TD3BCEPLearner", "model_config": get_td3bcep_mujoco_config()}
        ),
        "awac_mujoco": ml_collections.ConfigDict(
            {"model_constructor": "AWACLearner", "model_config": get_awac_mujoco_config()}
        ),
        "awacep_mujoco": ml_collections.ConfigDict(
            {"model_constructor": "AWACEPLearner", "model_config": get_awacep_mujoco_config()}
        ),
    }
    return possible_structures[config_string]
