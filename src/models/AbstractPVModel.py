from abc import ABC, abstractmethod

import torch


class AbstractPVModel(torch.nn.Module, ABC):

    @abstractmethod
    def forward(self, x): ...
