# Reinforcement Learning: Gymnasium Library

This repository contains implementations of reinforcement learning algorithms applied to environments from the [Gymnasium](https://gymnasium.farama.org/index.html) library (the maintained successor to OpenAI Gym).

## Environments

The following evironments are (or will be) used:
- [FrozenLake-v1](https://gymnasium.farama.org/environments/toy_text/frozen_lake/)
- [LunarLander-v2](https://gymnasium.farama.org/environments/box2d/lunar_lander/)

## Results

<div align="center">
<table>
  <tr>
    <th>Name</th>
    <th>Algorithm</th>
    <th>Solved Demo</th>
  </tr>
  <tr>
    <td><b><a href="https://proceedings.neurips.cc/paper_files/paper/2010/file/091d584fced301b442654dd8c23b3fc9-Paper.pdf" target="_blank"> DQL </a></b></td>
    <td><img src="assets/dql_algo.png" width="400"></td>
    <td><img src="assets/dql_frozen_lake.gif" width="300"></td>
  </tr>

  <tr>
    <td><b><a href="https://doi.org/10.48550/arXiv.1707.06347" target="_blank"> PPO Clip </a></b></td>
    <td><img src="assets/ppo_algo.png" width="400"></td>
    <td><img src="assets/ppo_lunar_lander.gif" width="300"></td>
  </tr>
</table>
</div>
