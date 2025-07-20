# Prism
Prism is a modular system of deep Q-learning algorithms and modifications that can be combined.

## Available Components
[IQN](https://arxiv.org/abs/1806.06923)\
[IDS](https://arxiv.org/abs/1812.07544)\
[DQN](https://arxiv.org/abs/1312.5602)\
[PER](https://arxiv.org/abs/1511.05952)\
[Double Q Learning](https://arxiv.org/abs/1509.06461)\
[Layer Normalization](https://arxiv.org/abs/1607.06450)\
[N-Step Returns](https://gibberblot.github.io/rl-notes/single-agent/n-step.html)

## Best Combination
The default configuration is IDS + IQN + Layer Normalization + 3-step returns with a target network. This seemed
to perform the best across the MinAtar experiments.

## Data
To find the best combination of the components available, I ran additive and subtractive ablation studies similar to
the [Revisiting Rainbow](https://arxiv.org/abs/2011.14826) paper. Data from the studies are available on WandB. 
The additive study is [here](https://wandb.ai/aech/Prism%20Subtractive%20Ablation%20Experiment?nw=nwuseraech)
and the subtractive study is [here](https://wandb.ai/aech/Second%20Prism%20Additive%20Ablation%20Experiment?nw=nwuseraech).
Due to time constraints, I only ran experiments in MinAtar. Hyperparameters can be found in the corresponding config 
and experiment files in this repo.