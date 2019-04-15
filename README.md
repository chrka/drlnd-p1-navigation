# DRLND Project 1: Navigation

<iframe width="560" height="315" src="https://www.youtube.com/embed/hucCBvA1qT8" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

The goal of the agent is to move around in the world trying to gather as many
yellow bananas during each episode, while simultaneously trying to avoid any blue bananas.

The agent receives information about its velocity and what objects can be found
in 36 directions around it, ie., it is dealing with a state space that has 
37 dimensions. It can take any of four possible discrete actions: move forward, move
backwards, turn left, or turn right. 

The agent receives a reward of +1 for every yellow banana, and -1 for each blue banana.
The agent is considered to have solved its task when it manages to get an average
score of at least +13 over 100 episodes. 

## Installation

Clone this repository and install the requirements needed as per the instructions below.

### Python Requirements

Follow the instructions in the Udacity [Deep Reinforcement Learning repository](https://github.com/udacity/deep-reinforcement-learning)
on how to set up the `drlnd` environment, and then also install the [Click](https://click.palletsprojects.com/en/7.x/)
package (used for handling command line arguments):
```shell
pip install click
```

Alternatively, on some systems it might be enough to install the required packages
from the provided `requirements.txt` file:
```shell
pip install -r requirements.txt
```

### Unity environment

Download the Unity environment appropriate for your operating system using the links below and unzip
it into the project folder.

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)


## Training and Running the Agent

To train the agent, use the `train.py` program which takes the Unity environment
and optional arguments to experiment with various parameters.

```shell
(drlnd) $ python train.py --help
Usage: train.py [OPTIONS]

Options:
  --environment PATH     Path to Unity environment  [required]
  --layer1 INTEGER       Number of units in input layer
  --layer2 INTEGER       Number of units in hidden layer
  --eps-decay FLOAT      Epsilon decay factor
  --eps-min FLOAT        Minimum value of epsilon
  --plot-output PATH     Output file for score plot
  --weights-output PATH  File to save weights to after success
  --seed INTEGER         Random seed
  --help                 Show this message and exit.
```

The default values are:

| Option | Value |
|--------|-------|
|layer1 | 32 |
|layer2 | 32 |
|eps-decay| 0.999 |
|eps-min | 0.01 |
|plot-output | score.png |
|weights-output | weights.pth |
|seed | None — do not set | 

For example:

```shell
(drlnd) $ python train.py --environment=Banana.app --seed=20190415 
```

After successfully training the agent, use the `run.py` program to load
the weights and run the simulation, which takes similar parameters as
the training program:

```shell
(drlnd) $ python run.py --help
Usage: run.py [OPTIONS]

Options:
  --environment PATH    Path to Unity environment  [required]
  --layer1 INTEGER      Number of units in input layer
  --layer2 INTEGER      Number of units in hidden layer
  --n-episodes INTEGER  Number of episodes to run
  --weights-input PATH  Network weights
  --help                Show this message and exit.
```

Default values for running the agent are:

| Option | Value |
|--------|-------|
|layer1 | 32 |
|layer2 | 32 |
|n-episodes | 3|
|weights-input| weights.pth | 

For example:
```
(drlnd) $ python run.py --environment=Banana.app --weights-input=weights.pth
```