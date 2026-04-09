from collections import deque
import random
import numpy as np

from .Game import BOARD_SIZE


def apply_symmetry(x: np.ndarray, pi: np.ndarray):
    """The board has 6 (?) different symmetries, after which
    the reasoning remains exactly the same. This function generates
    new symmatrical boards to augment the data diversity seen
    during training.

    Args:
        x (np.ndarray): Board, shape (2, 4, 4, 4)
        pi (np.ndarray): Policy probabilities, shape (16,)

    Returns:
        tuple[np.ndarray, np.ndarray]: Board and policy probabilities, after applying a random symmetry.
    """
    x_augmented = np.asarray(x, dtype=np.float32)
    pi_augmented = np.asarray(pi, dtype=np.float32).reshape(BOARD_SIZE, BOARD_SIZE)

    if random.random() < 0.5:
        x_augmented = np.flip(x_augmented, axis=1)
        pi_augmented = np.flip(pi_augmented, axis=0)

    rotation_count = random.randrange(4)
    if rotation_count:
        x_augmented = np.rot90(
            x_augmented,
            k=rotation_count,
            axes=(1, 2),
        )
        pi_augmented = np.rot90(
            pi_augmented,
            k=rotation_count,
            axes=(0, 1),
        )

    return (
        np.ascontiguousarray(x_augmented, dtype=np.float32),
        np.ascontiguousarray(
            pi_augmented.reshape(BOARD_SIZE**2),
            dtype=np.float32,
        ),
    )


class ReplayBuffer:
    """
    A memory buffer to remember moves, their outcomes,
    and serve them back to train a model.
    """

    def __init__(self, capacity: int):
        self.data = deque(maxlen=capacity)

    def __len__(self):
        return len(self.data)

    def push_example(self, example):
        self.data.append(example)

    def push_examples(self, examples):
        for example in examples:
            self.push_example(example)

    def sample_batch(self, batch_size: int):
        assert (
            batch_size <= len(self.data)
        ), "Not enough examples in the buffer to sample a batch."
        batch = random.sample(self.data, batch_size)
        batch = [apply_symmetry(b[0], b[1]) for b in batch]

        x_batch = np.stack([b[0] for b in batch])
        pi_batch = np.stack([b[1] for b in batch])
        z_batch = np.stack([b[2] for b in batch])

        return x_batch, pi_batch, z_batch
