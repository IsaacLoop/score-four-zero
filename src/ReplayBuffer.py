from collections import deque
import numpy as np


class ReplayBuffer:

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
        indices = np.random.choice(len(self.data), size=batch_size, replace=False)
        batch = [self.data[i] for i in indices]

        x_batch = np.stack([b[0] for b in batch])
        pi_batch = np.stack([b[1] for b in batch])
        z_batch = np.stack([b[2] for b in batch])

        return x_batch, pi_batch, z_batch
