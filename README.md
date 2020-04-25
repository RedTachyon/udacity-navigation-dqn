# udacity-navigation-dqn

This is my implementation of the first Nanodegree project about navigation.

I will assume setup has been performed according to the instructions (i.e. Python 3.6 with PyTorch and UnityAgents installed)

The environment used here is the banana collection environment, with states represented as 37-dimensional vectors, and 4 discrete actions. The agent's task is to navigate through the environment and collect yellow bananas, while avoiding purple ones.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the unzipped file in this directory

### Instructions

Use the `Navigation.ipynb` notebook to train and evaluate the agent.

### Solving criterion

The original formulation of the problem states that it's considered solved when the
100-episode average reward is at least 13. For robustness, I increased that value to 15, since it's still attained
in a fairly short time.