import numpy as np

from typing import Optional, Tuple

from src.core import Tensor, Layer


class Flatten(Layer):
    def __init__(self, name: Optional[str] = None):
        super().__init__()
        self.name = name if name else f"flatten_{id(self)}"

    def forward(self, input_tensor: Tensor) -> Tensor:
        original_input_shape = input_tensor.shape
        
        if input_tensor.ndim < 2:
            raise ValueError(f"{self.name} expects input with at least 2 dimensions (N, ...), "
                             f"got {input_tensor.ndim}D shape {input_tensor.shape}")

        batch_size = input_tensor.shape[0]
        num_features = np.prod(input_tensor.shape[1:])
        output_data = input_tensor.data.reshape(batch_size, num_features)
        
        res_tensor = Tensor(output_data, [input_tensor], self.name)
        
        def __backward():
            if not input_tensor.requires_grad:
                return
            
            input_tensor.gradient += res_tensor.gradient.reshape(original_input_shape)
            
        res_tensor._Tensor__backward = __backward
        self.output = res_tensor

        return res_tensor