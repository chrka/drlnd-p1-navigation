from collections import deque

import numpy as np
import matplotlib.pyplot as plt
import torch
from unityagents import UnityEnvironment

from agent import Agent


def dqn(env, agent: Agent, n_episodes=3000, eps_start=1.0, eps_min=0.01,
        eps_decay=0.999):
    # Assume we're operating brain 0
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    action_size = brain.vector_action_space_size

    scores = []
    score_window = deque(maxlen=100)
    eps = eps_start
    for i in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        while True:
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        score_window.append(score)
        scores.append(score)
        eps = max(eps_min, eps_decay * eps)
        line_end = "\n" if i % 100 == 0 else ""
        print(f"\rEpisode {i}\tAverage score {np.mean(score_window):.2f}",
              end=line_end)
        if np.mean(score_window) > 13.0:
            print(f"\nEnvironment solved in {i} episodes.")
            break
    return scores


if __name__ == '__main__':
    env = UnityEnvironment(file_name="Banana.app")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent = Agent(state_size=37, action_size=4, device=device, layer1=128,
                  layer2=96)

    scores = dqn(env, agent)

    env.close()

    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.show()