import numpy as np

from typing import Optional,  List

from src.core import Tensor, Layer
from src.functions import ActivationFunction


class Dense(Layer):
    def __init__(self, units: int, activation_class: Optional[type[ActivationFunction]] = None, 
                 use_bias: bool = True, name: Optional[str] = None):
        
        super().__init__() 
        self.units = units
        self.activation_class = activation_class
        self.use_bias = use_bias
        self.name = name if name else f"dense_{id(self)}"

        self.kernel: Optional[Tensor] = None
        self.bias: Optional[Tensor] = None
        self._input_features: Optional[int] = None
        self._weights_initialized: bool = False
        self.weights: List[Tensor] = [] 

    def _ensure_weights_initialized(self, input_features: int):
        if self._weights_initialized and self._input_features == input_features:
            return
        
        self._input_features = input_features
        
        limit = np.sqrt(6.0 / (self._input_features + self.units)) 
        kernel_data = np.random.uniform(-limit, limit, size=(self._input_features, self.units)).astype(float)
        self.kernel = Tensor(kernel_data, tensor_type=f"{self.name}_kernel")
        
        current_params = [self.kernel] 

        if self.use_bias:
            bias_data = np.zeros((1, self.units), dtype=float) 
            self.bias = Tensor(bias_data, tensor_type=f"{self.name}_bias")
            current_params.append(self.bias)
        
        self.weights = current_params 
        self._weights_initialized = True

    def set_weights_from_keras(self, keras_weights_list: List[np.ndarray]):
        if not (1 <= len(keras_weights_list) <= 2):
            raise ValueError(f"Dense layer {self.name} expects 1 or 2 weight arrays, got {len(keras_weights_list)}")

        k_np = keras_weights_list[0].astype(float)
        input_features_from_kernel = k_np.shape[0]
        units_from_kernel = k_np.shape[1]

        if self.units != units_from_kernel:
            raise ValueError(f"Units mismatch for {self.name}. Configured: {self.units}, from kernel: {units_from_kernel}")

        self._ensure_weights_initialized(input_features_from_kernel) 

        if self.kernel.shape != k_np.shape:
             raise ValueError(f"Kernel shape mismatch for {self.name}. Expected {self.kernel.shape}, got {k_np.shape}")
        self.kernel.data = k_np
        self.kernel.gradient.fill(0.0)

        if self.use_bias:
            if len(keras_weights_list) == 2:
                b_np = keras_weights_list[1].astype(float).reshape(1, self.units) 
                if self.bias.shape != b_np.shape: 
                    raise ValueError(f"Bias shape mismatch for {self.name}. Expected {self.bias.shape}, got {b_np.shape} from Keras {keras_weights_list[1].shape}")
                self.bias.data = b_np
                self.bias.gradient.fill(0.0)
            else: 
                print(f"Warning: Dense layer {self.name} uses bias, but Keras weights list had no bias. Bias remains as initialized.")
        elif len(keras_weights_list) == 2: 
             print(f"Warning: Dense layer {self.name} has use_bias=False, but Keras weights included bias. It was ignored.")
        
        self._weights_initialized = True 

    def forward(self, input_tensor: Tensor) -> Tensor:
        if input_tensor.ndim != 2: 
            raise ValueError(f"{self.name} expects 2D input (N, input_features), got {input_tensor.ndim}D.")
        
        N, input_features = input_tensor.shape
        self._ensure_weights_initialized(input_features) 

        output_before_activation = input_tensor @ self.kernel 
        
        if self.use_bias:
            output_before_activation = output_before_activation + self.bias 
        if self.activation_class:
            final_output = output_before_activation.compute_activation(self.activation_class)
        else:
            final_output = output_before_activation
        
        self.output = final_output
        return final_output

    def get_parameters(self) -> List[Tensor]:
        return self.weights