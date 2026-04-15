# Run 2

## Many experiments

I have tried a few different configurations before starting this final run, and made them fight against each other to know which approach was more promising. I put some checkpoints from run-1 in the pool as well. Here are the final results, after ~1000 fights each, with elo scores (they all started with the same elo):

| Rank | Model | Elo |
| --- | --- | ---: |
| 1 | 5conv4M it. 1000 | 914.9 |
| 2 | 2conv600k it. 3507 | 906.4 |
| 3 | 3conv1M it. 1000 | 888.7 |
| 4 | 6conv4M it. 954 | 880.5 |
| 5 | run-1 it. 8880 | 404.5 |
| 6 | run-1 it. 9990 | 379.5 |
| 7 | run-1 it. 7730 | 257.8 |
| 8 | run-1 it. 5400 | 255.9 |
| 9 | run-1 it. 6480 | 251.9 |
| 10 | run-1 it. 2170 | 215.2 |
| 11 | run-1 it. 620 | 144.6 |

"`5conv4M it. 1000`" is a 3D CNN with a depth of 5, around 4 million trainable parameters, and it is the model as it was after 1000 iterations of training. That number of iterations does not mean that much since training steps per iterations were not fixed during my experiments, but what this ranking allowed me to see is that depth and parameter count don't matter that much, since a model extremely similar in architecture to that of run-1 ended up in 2nd place, even though its training run wasn't even over. The real difference is in training steps, and new fresh self-play data. Those are the things that I have really been scaling for run-2.

I also experimented with an alternative architecture with a transformers-based backbone, but it didn't perform as well, and was much slower to train.

Since the board has many symmetries, I have been using a form of data augmentation, at the replay buffer level, where I randomly rotate the board by 0/90/180/270 degrees, or mirror it, before feeding the training batches to the model.

This is why I have decided to have as final run-2 run, a network slightly deeper than the one in run-1 (`2` -> `3`), with more parameters (`0.59M` -> `2.57M`), and finally **much more** self-play games per iteration (`32` -> `768`) and training steps per iteration (`128` -> `768`), which are themselves each comprised of much bigger batches (`128` -> `1,536`).

Therefore, over the course of the `10,000` iterations of run-2, the model will have seen `24x` more self-play games, and been trained on `72x` more training steps, than in run-1. 

## Configuration

- Architecture: 3D CNN policy-value network
- CNN depth: `3` convolutional layers
- Approximate size: `2.57M` parameters
- Backbone shape: `2 -> 24 -> 48 -> 96` channels, then a shared fully connected layer of width `384`
- Training duration: `10,000` iterations
- Self-play games per iteration: `768`
- MCTS simulations per move: `100`
- Training steps per iteration: `768`
- Batch size: `1,536`
- Replay buffer size: `1,500,000`
- Optimizer: `AdamW`
- Learning-rate schedule: linear warmup over about `10` iterations, then cosine decay from `5e-4` to `0`
