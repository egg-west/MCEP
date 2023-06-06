#! /usr/bin/env python
import gym
import jax
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from MCEP.agents import TD3BCLearner
from MCEP.data import D4RLDataset
from MCEP.evaluation import evaluate, evaluate_normalized_state
from MCEP.wrappers import wrap_gym

FLAGS = flags.FLAGS

SEEDS = [11111]
SEED = SEEDS[0]

flags.DEFINE_string("env_name", "halfcheetah-medium-expert-v2", "Environment name.")
flags.DEFINE_integer("seed", SEED, "Random seed.")
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 5000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(1e6), "Number of training steps.")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_float("filter_percentile", None, "Take top N% trajectories.")
flags.DEFINE_float(
    "filter_threshold", None, "Take trajectories with returns above the threshold."
)
config_flags.DEFINE_config_file(
    "config",
    "configs/offline_config.py:td3bc_mujoco",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


def main(_):
    wandb.init(project=f"MCEP_offline_td3bc-{FLAGS.config.model_config.alpha}",
               group="td3bc-"+FLAGS.env_name,
               name = str(FLAGS.seed))
    wandb.config.update(FLAGS)

    env = gym.make(FLAGS.env_name)
    env = wrap_gym(env)
    env.seed(FLAGS.seed)

    dataset = D4RLDataset(env)
    state_mean, state_std = dataset.normalize_state()

    if FLAGS.filter_percentile is not None or FLAGS.filter_threshold is not None:
        dataset.filter(
            percentile=FLAGS.filter_percentile, threshold=FLAGS.filter_threshold
        )
    dataset.seed(FLAGS.seed)

    if "antmaze" in FLAGS.env_name:
        dataset.dataset_dict["rewards"] *= 100
    elif FLAGS.env_name.split("-")[0] in ["hopper", "halfcheetah", "walker2d"]:
        dataset.normalize_returns(scaling=1000)

    kwargs = dict(FLAGS.config.model_config)
    if kwargs.pop("cosine_decay", False):
        kwargs["decay_steps"] = FLAGS.max_steps
    agent = globals()[FLAGS.config.model_constructor](
        FLAGS.seed, env.observation_space, env.action_space, **kwargs
    )

    for i in tqdm.tqdm(
        range(1, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm
    ):
        batch = dataset.sample(FLAGS.batch_size)
        info = agent.update(batch, i % kwargs["policy_delay"] == 0)

        if i % FLAGS.log_interval == 0:
            info = jax.device_get(info)
            wandb.log(info, step=i)

        if i % FLAGS.eval_interval == 0:
            eval_info = evaluate_normalized_state(agent, env, FLAGS.eval_episodes, state_mean.squeeze(0), state_std.squeeze(0))
            #eval_info = evaluate(agent, env, FLAGS.eval_episodes)
            eval_info["return"] = env.get_normalized_score(eval_info["return"]) * 100.0
            for k, v in eval_info.items():
                wandb.log({f"evaluation/{k}": v}, step=i)


if __name__ == "__main__":
    app.run(main)
