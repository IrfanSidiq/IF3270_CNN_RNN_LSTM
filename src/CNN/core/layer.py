from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from ..core import Tensor


class Layer(ABC):
    """
    Abstract base class for all layers.
    """
    def __init__(self, name: Optional[str] = None):
        self.weights: List[Tensor] = []
        self.output: Optional[Tensor] = None
        self.name = name if name else self.__class__.__name__.lower() + f"_{id(self)}"
        
        self._weights_initialized: bool = False 
        self._built: bool = False 
        self._input_shape_to_layer: Optional[Tuple[int, ...]] = None 

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