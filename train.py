import random
from collections import deque

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
from unityagents import UnityEnvironment

from agent import Agent


def train(env, agent, weight_path, n_episodes=3000, eps_start=1.0,
          eps_min=0.01,
          eps_decay=0.999, threshold=13.0):
    """Train agent and store weights if successful.

    Args:
        env (UnityEnvironment): Environment to train agent in
        agent (Agent): Agent to train
        weight_path (str): Path for weights file
        n_episodes (int): Max number of episodes to train agent for
        eps_start (float): Initial epsilon value
        eps_min (float): Minimum value of epsilon value
        eps_decay (float): Epsilon decay factor
        threshold (float): Min mean score over 100 episodes consider success
    """
    # Assume we're operating brain 0
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

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
        print(f"\rEpisode {i:4d}\tAverage score {np.mean(score_window):.2f}",
              end="\n" if i % 100 == 0 else "")
        if len(score_window) >= 100 and np.mean(score_window) > threshold:
            print(f"\nEnvironment solved in {i} episodes.")
            agent.save_weights(weight_path)
            break
    return scores


@click.command()
@click.option('--environment', required=True,
              help="Path to Unity environment", type=click.Path())
@click.option('--layer1', default=32, help="Number of units in input layer")
@click.option('--layer2', default=32, help="Number of units in hidden layer")
@click.option('--eps-decay', default=0.999, help="Epsilon decay factor")
@click.option('--eps-min', default=0.01, help="Minimum value of epsilon")
@click.option('--plot-output', default="score.png",
              help="Output file for score plot", type=click.Path())
@click.option('--weights-output', default='weights.pth',
              help="File to save weights to after success", type=click.Path())
@click.option('--seed', type=int, help="Random seed")
def main(environment, layer1, layer2, eps_decay, eps_min,
         plot_output, weights_output, seed):
    # Set seed if given to help with reproducibility
    if seed:
        print(f"Using seed {seed}")
        random.seed(seed)
        torch.random.manual_seed(seed)

    # Initialize Unity environment from external file
    env = UnityEnvironment(file_name=environment)

    # Use CUDA if available, cpu otherwise
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create agent given model parameters
    agent = Agent(state_size=37, action_size=4, device=device, layer1=layer1,
                  layer2=layer2)

    # Train agent (will save weights if successful)
    scores = train(env, agent, weights_output,
                   eps_decay=eps_decay, eps_min=eps_min)

    env.close()

    # Generate score plot
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.savefig(plot_output)


if __name__ == '__main__':
    main()
