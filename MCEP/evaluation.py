from typing import Dict

import gym
import numpy as np

from gym.wrappers.monitoring.video_recorder import VideoRecorder


from MCEP.data.dataset import Dataset


def evaluate(agent, env: gym.Env, num_episodes: int) -> Dict[str, float]:
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=num_episodes)
    for _ in range(num_episodes):
        observation, done = env.reset(), False
        while not done:
            action = agent.eval_actions(observation)
            observation, _, done, _ = env.step(action)

    return {"return": np.mean(env.return_queue), "length": np.mean(env.length_queue)}

def evaluate_normalized_state(agent, env: gym.Env,
                              num_episodes: int,
                              state_mean: float,
                              state_std: float) -> Dict[str, float]:
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=num_episodes)
    for _ in range(num_episodes):
        observation, done = env.reset(), False
        observation = (observation - state_mean)/state_std
        while not done:
            action = agent.eval_actions(observation)
            observation, _, done, _ = env.step(action)
            observation = (observation - state_mean)/state_std

    return {"return": np.mean(env.return_queue), "length": np.mean(env.length_queue)}

def evaluate_normalized_state_latent_action(agent, env: gym.Env,
                                            num_episodes: int,
                                            state_mean: float = 0.0,
                                            state_std: float = 1.0) -> Dict[str, float]:
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=num_episodes)
    for _ in range(num_episodes):
        observation, done = env.reset(), False
        observation = (observation - state_mean)/state_std
        while not done:
            action = agent.eval_latent_actions(observation)
            observation, _, done, _ = env.step(action)
            observation = (observation - state_mean)/state_std

    return {"return": np.mean(env.return_queue), "length": np.mean(env.length_queue)}

def evaluate_surrogate_normalized_state(agent, env: gym.Env,
                              num_episodes: int,
                              state_mean: float,
                              state_std: float) -> Dict[str, float]:
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=num_episodes)
    for _ in range(num_episodes):
        observation, done = env.reset(), False
        observation = (observation - state_mean)/state_std
        while not done:
            action = agent.eval_surrogate_actions(observation)
            observation, _, done, _ = env.step(action)
            observation = (observation - state_mean)/state_std

    return {"return": np.mean(env.return_queue), "length": np.mean(env.length_queue)}

def evaluate_surrogate(agent, env: gym.Env,
                              num_episodes: int) -> Dict[str, float]:
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=num_episodes)
    for _ in range(num_episodes):
        observation, done = env.reset(), False

        while not done:
            action = agent.eval_surrogate_actions(observation)
            observation, _, done, _ = env.step(action)

    return {"return": np.mean(env.return_queue), "length": np.mean(env.length_queue)}

def evaluate_normalized_state_video(agent, env: gym.Env,
                              num_episodes: int,
                              state_mean: float,
                              state_std: float,
                              video_name: str) -> Dict[str, float]:
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=num_episodes)
    video = VideoRecorder(env, video_name, enabled=True)
    for _ in range(num_episodes):
        observation, done = env.reset(), False
        observation = (observation - state_mean)/state_std
        while not done:
            action = agent.eval_actions(observation)
            observation, _, done, _ = env.step(action)
            env.render()
            video.capture_frame()
            observation = (observation - state_mean)/state_std
    video.close()
    return {"return": np.mean(env.return_queue), "length": np.mean(env.length_queue)}


def evaluate_log_prob(agent, dataset: Dataset, batch_size: int = 2048) -> float:
    num_iters = len(dataset) // batch_size
    total_log_prob = 0.0
    for j in range(num_iters):
        indx = np.arange(j * batch_size, (j + 1) * batch_size)
        batch = dataset.sample(batch_size, keys=("observations", "actions"), indx=indx)
        log_prob = agent.eval_log_probs(batch)
        total_log_prob += log_prob

    return total_log_prob / num_iters