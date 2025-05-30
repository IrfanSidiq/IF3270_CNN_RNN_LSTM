from abc import ABC, abstractmethod
from typing import List, Optional

from src.core import Tensor


class Layer(ABC):
    """
    Abstract base class for all layers.
    """
    def __init__(self):
        self.weights: List[Tensor] = []
        self.output: Optional[Tensor] = None

    @abstractmethod
    def forward(self, input: Tensor) -> Tensor:
        """
        Performs the forward pass of the layer.
        """
        pass

    def get_parameters(self) -> List[Tensor]:
        """
        Returns all learnable parameters (Tensors) of this layer.
        """
        return self.weights