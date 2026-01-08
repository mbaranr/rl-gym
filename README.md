# Reinforcement Learning: Gymnasium Library

This repository showcases implementations of reinforcement learning algorithms applied to two environments of the OpenAI’s Gym Library: "*Lunar Lander*" and "*Frozen Lake*"

## LunarLander

Classic rocket trajectory optimization problem. According to Pontryagin’s maximum principle, it is optimal to fire the engine at full throttle or turn it off. This is the reason why this environment has **discrete** actions: engine on or off ([Gymnasium](https://gymnasium.farama.org/environments/box2d/lunar_lander/#lunar-lander)).

### PPO Clip

How can we take the biggest possible improvement step on a policy using the data we currently have, without stepping so far that we accidentally cause performance collapse?

<div align="center">
  <img src="assets/ppo_algo.png" width="400"/>
</div>

### Results

<div align="center">
  <img src="https://github.com/M4mbo/PPO_Clip_for_LunarLander/assets/115642529/670b9f08-b424-4c43-acb9-98abe4aefbc1" width="400"/>
</div>

## Frozen Lake

Frozen lake involves crossing a frozen lake from start to goal without falling into any holes by walking over the frozen lake. The player may not always move in the intended direction due to the slippery nature of the frozen lake ([Gymnasium](https://gymnasium.farama.org/environments/box2d/lunar_lander/#lunar-lander)).

### Double Deep Q-Learning

Double Deep Q-Learning is used to reduce the maximaztion bias in Q-Learning. This entails using two separate Q-value estimators, each of which is used to update the other. The target values are calculated using a target Q-network. The target Q-network's parameters are updated to the current networks every $C$ time steps.


<div align="center">
  <img src="assets/ddql_algo.png" width="400"/>
</div>

### Results

<div align="center">
  <img src="assets/frozen_lake_demo.gif" width="300"/>
</div>

### Credits

* [mbaranr](https://github.com/mbaranr) - Code.
* [LMH](https://www.lmh.ox.ac.uk/) summer program on 'AI and ML: Advanced Applications' - Theory.
