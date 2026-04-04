# Score Four Zero

AlphaZero-style self-play RL agent for "Score Four" (a kind of 4×4×4 3D "Connect Four").

<img src="img/game_illustration.png" alt="alt text" style="height:300px;" />

## Rules of the game

Two players face each other. White begins by placing a bead in any of the 16 rods. Then black does the same. Then white, etc. Each rod has a maximum height of 4 beads.

__The first one to align 4 beads of their own color wins__. Aligning can be horizontal, vertical, or diagonal. There are 76 ways to win.

If no player manages to have aligned 4 beads when the board is full, then it's a __draw__.

## Goals of the project

My ambition is to train an agent to play Score Four better than any human ever could, using the same principles as that of AlphaZero (self-play, Monte Carlo Tree Search, and a _kinda deep_ neural network).

## Environment

To replicate the project on your machine, you need to install the Python environment. Find more detailed instructions in [env/readme.md](env/readme.md).

## Training the agent

The training script is `src/train_CLI.py`. In the environment, you start a training run with:

```bash
python -m src.train_CLI
```

Optional and mutually exclusive flags are:
- `--resume`: resume training from the latest checkpoint (as long as it appears that the previous run was made with the same training hyperparameters as the ones currently set in the code)
- `--delete-existing-checkpoints`: delete all existing checkpoints before starting training

During training, you can monitor metrics such as the losses, the learning rate, the memory size, etc.with TensorBoard.

In your environment, run:

```bash
tensorboard --logdir tb_logs
```

and open the provided local URL in your browser.

## Evaluating the agent

The evaluation script is `src/evaluate_CLI.py`. In the environment, you start an evaluation run with:

```bash
python -m src.evaluate_CLI
```

This script will automatically pick up all model checkpoints in `checkpoints/` and make them play against each other, _forever_, and rank them using an [Elo rating system](https://en.wikipedia.org/wiki/Elo_rating_system). At the end of each game between two models, their Elo ratings are updated.

All of this can be monitored in real time using a local server that runs with the script.

## Playing against the agent

The play script is `src/play_CLI.py`. In the environment, you start a play run with:

```bash
python -m src.play_CLI
```

It automatically picks up the most advanced checkpoint in `checkpoints/` and lets you play against it in TUI. It is however possible to select a different checkpoint with the `--checkpoint` flag.

## My mindset regarding AI assisted coding in this repository

This repository is made both for me to showcase my capabilities as a data scientist and for me to deeply understand something new. Hence, every part of the code that is critical to the ML concepts is of course hand-coded by me, without any AI assistance. I did, however, use GPT-5.4 through OpenAI's Codex to write some non-critical parts of the project, such as the Elo visualization server and some optimizations for parallelizing some stuff. None of that was beyond the reach of a motivated developer, but those non-ML aspects just weren't the point of the project. Anything made with AI is advertised as such in the codebase.

## Observations

The agent extremely quickly reaches a state where it is seemingly unbeatable by me. As of right now, I have not yet seen declining rates of improvement, but I have only trained it for a few hours. I will update this section as I train it more and see how it evolves. It is fascinating to see that the elo ratings of the checkpoints seems to increase linearly with the duration of training. It just seems remarkably stable and predictable, which is not that common in deep RL.

Making a true pipeline training an agent using self-play / MCTS / deep learning should not be that hard. This very humble project is a great proof of that. All of the critical and somewhat complex code is located in `src/MCTS.py` and in `src/train_CLI.py`, which total less than 500 uncompacted lines of code. I found a terrible lack of clear explanations and resources to replicate AlphaZero-like projects, so I hope that this repository can be useful to _you_.

As of right now, almost all of the hyperparameters used for training (number of iterations, games per iteration, training cycles per iteration, depth of tree exploration, size of the model, learning rate, learning rate scheduling, etc) have been chosen arbitrarily, based on my feeling of what would be reasonable, and worked well enough. That is to say, a great deal of optimization is probably still possible.