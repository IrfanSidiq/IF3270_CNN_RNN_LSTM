from typing import List, Optional

from src.core import Tensor
from src.functions import LossFunction
from src.layers import Layer
from src.optimization import Optimizer


class Sequential:
    def __init__(self, layers: Optional[List[Layer]] = None):
        """
        Initializes a Sequential model with given list of Layer objects.
        """
        self.layers: List[Layer] = []
        if layers:
            for layer in layers:
                self.add(layer)

        self.optimizer: Optimizer = None
        self.loss_function: LossFunction = None

    def add(self, layer: Layer) -> None:
        """
        Adds a layer to the model.
        """
        if not isinstance(layer, Layer):
            raise TypeError(f"Expected Layer instance, but got {type(layer).__name__}")
        
        self.layers.append(layer)

    def forward(self, input_tensor: Tensor) -> Tensor:
        """
        Performs a forward pass through all layers in the model.
        """
        if not self.layers:
            raise ValueError("Cannot perform forward pass on an empty model.")

        x = input_tensor
        for layer in self.layers:
            x = layer.forward(x)
            
        return x

    def get_parameters(self) -> List[Tensor]:
        """
        Retrieves all trainable parameters from all layers in the model.
        """
        params: List[Tensor] = []
        for layer in self.layers:
            layer_params = layer.get_parameters()
            for p in layer_params:
                if p.requires_grad:
                    params.append(p)

        return params