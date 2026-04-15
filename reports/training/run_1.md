# Run 1

Run-1 is the first completed long training campaign in this repository.

## Configuration

- Architecture: 3D CNN policy-value network
- CNN depth: `2` convolutional layers
- Approximate size: `592k` parameters
- Backbone shape: `2 -> 32 -> 64` channels, then a shared fully connected layer of width `128`
- Training duration: `10,000` iterations
- Self-play games per iteration: `32`
- MCTS simulations per move: `100`
- Training steps per iteration: `128`
- Batch size: `128`
- Replay buffer size: `100,000`
- Optimizer: `AdamW`
- Learning-rate schedule: cosine decay from `3e-4` to `3e-5`

## Training signals

Policy loss:

<img src="../img/runs/big_run_1/loss_policy.png" alt="Run-1 policy loss" style="height:300px;" />

Value loss:

<img src="../img/runs/big_run_1/loss_value.png" alt="Run-1 value loss" style="height:300px;" />

Win rate against previous checkpoints:

<img src="../img/runs/big_run_1/checkpoint_winrate.png" alt="Run-1 checkpoint win rate" style="height:300px;" />

The checkpoint win-rate curves stayed above `0.5` against checkpoints from `20`, `50`, and `100` iterations earlier, which was the strongest sign that the run kept improving across the full training window.

One of the more interesting run-1 observations was that the value loss kept going up while the policy loss only really started going down midway through training. That did not appear to mean the model was getting worse. A more plausible interpretation was that, as the model improved, it kept filling its replay memory with stronger and more complex self-play positions, which simply made the training targets harder. The checkpoint win-rate curves were the clearest signal that the run was still progressing well from beginning to end.

## Post-training checkpoint ranking

During training, the model was saved at regular intervals. After training, all of those checkpoints were ranked against each other with `python -m src.checkpoint_ranker_CLI`.

<img src="../img/runs/big_run_1/checkpoint_ranks.png" alt="Run-1 checkpoint Elo ranking" style="height:450px;" />

### Observations 

The agent extremely quickly reaches a state where it becomes quite challenging for me. I can still beat it with some effort, but it already feels like an okay-ish opponent. As of right now, I have not yet seen declining rates of improvement, but I have only trained it for a few hours. I will update this section as I train it more and see how it evolves. It is fascinating to see that the elo ratings of the checkpoints seems to increase linearly with the duration of training. It just seems remarkably stable and predictable, which is not that common in deep RL.

Making a true pipeline training an agent using self-play / MCTS / deep learning should not be that hard. This very humble project is a great proof of that. All of the critical and somewhat complex code is located in src/MCTS.py and in src/train_CLI.py, which total less than 500 uncompacted lines of code. I found a terrible lack of clear explanations and resources to replicate AlphaZero-like projects, so I hope that this repository can be useful to you.

As of right now, almost all of the hyperparameters used for training (number of iterations, games per iteration, training cycles per iteration, depth of tree exploration, size of the model, learning rate, learning rate scheduling, etc) have been chosen arbitrarily, based on my feeling of what would be reasonable, and worked well enough. That is to say, a great deal of optimization is probably still possible.

### Next steps as they were framed after run-1

- Use a few models from that first serious batch as Elo anchors for future evaluations.
- Train longer and find the real plateau.
- Make the model bigger.
- Increase the number of MCTS simulations.
- etc.