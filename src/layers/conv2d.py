import numpy as np

from typing import Union, Tuple, Optional, List

from src.core import Tensor, Layer
from src.functions import ActivationFunction, ReLU
from src.utils import convolve2d


class Conv2D(Layer):
    def __init__(self,
        num_kernels: int,
        kernel_size: Union[int, Tuple[int, int]],
        input_channels: Optional[int] = None,
        strides: Union[int, Tuple[int, int]] = (1, 1),
        padding: str = 'valid',
        activation: Optional[Union[str, ActivationFunction]] = None,
        use_bias: bool = True,
        name: Optional[str] = None):
        
        super().__init__()
        self.num_kernels = num_kernels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        if len(self.kernel_size) != 2:
            raise ValueError("kernel_size must be an int or a tuple of 2 ints.")
            
        self.input_channels = input_channels
        self.strides = (strides, strides) if isinstance(strides, int) else strides
        if len(self.strides) != 2:
            raise ValueError("strides must be an int or a tuple of 2 ints.")

        if padding.lower() not in ['valid', 'same']:
            raise ValueError("padding must be 'valid' or 'same'.")
        self.padding = padding.lower()
        
        self.use_bias = use_bias
        self.name = name if name else f"conv2d_layer_{id(self)}"

        self.kernel: Optional[Tensor] = None
        self.bias: Optional[Tensor] = None
        self._weights_initialized: bool = False 

        if isinstance(activation, str):
            if activation.lower() == 'relu':
                self.activation_fn_class = ReLU
            else:
                self.activation_fn_class = None
        elif isinstance(activation, type) and issubclass(activation, ActivationFunction):
            self.activation_fn_class = activation
        elif activation is None:
            self.activation_fn_class = None
        else:
            raise TypeError(f"Unsupported activation type: {type(activation)}. "
                            "Expected str, ActivationFunction class, or None.")

    def _ensure_weights_initialized(self, C_in_from_input: int):
        """
        Initializes kernel and bias with zeros if they aren't already.
        This method is called from forward() if weights aren't set.
        """
        if self._weights_initialized:
            if self.input_channels is not None and self.input_channels != C_in_from_input:
                raise ValueError(
                    f"Layer {self.name} was configured/loaded with {self.input_channels} input channels, "
                    f"but received input with {C_in_from_input} channels."
                )
            return

        if self.input_channels is None:
            self.input_channels = C_in_from_input
        elif self.input_channels != C_in_from_input:
             raise ValueError(
                f"Layer {self.name} was configured with input_channels={self.input_channels} "
                f"but received first input with {C_in_from_input} channels."
            )

        KH, KW = self.kernel_size
        kernel_shape = (self.num_kernels, self.input_channels, KH, KW)
        kernel_data_np = np.zeros(kernel_shape, dtype=float)
        self.kernel = Tensor(kernel_data_np, tensor_type=f"{self.name}_kernel")
        
        self.weights = [self.kernel]

        if self.use_bias:
            bias_shape = (self.num_kernels,)
            bias_data_np = np.zeros(bias_shape, dtype=float)
            self.bias = Tensor(bias_data_np, tensor_type=f"{self.name}_bias")
            self.weights.append(self.bias)
        
        self._weights_initialized = True

    def set_weights_from_keras(self, keras_weights: List[np.ndarray]):
        """
        Sets the kernel and bias from weights in Keras format.
        Keras kernel shape: (kernel_height, kernel_width, input_channels, num_kernels)
        Keras bias shape: (num_kernels,)
        """
        if not (1 <= len(keras_weights) <= 2):
            raise ValueError(f"Expected 1 or 2 weight arrays (kernel, [bias]), got {len(keras_weights)}.")

        keras_kernel_np = keras_weights[0]
        if keras_kernel_np.ndim != 4:
            raise ValueError(f"Keras kernel must be 4D, got {keras_kernel_np.ndim}D.")

        # Keras kernel: (KH, KW, C_in, C_out/num_kernels) --> transpose to our kernel format: (C_out/num_kernels, C_in, KH, KW)
        kernel_np = np.transpose(keras_kernel_np, (3, 2, 0, 1)).astype(float)

        loaded_num_kernels = kernel_np.shape[0]
        loaded_input_channels = kernel_np.shape[1]
        loaded_kh, loaded_kw = kernel_np.shape[2], kernel_np.shape[3]

        if self.num_kernels != loaded_num_kernels:
            raise ValueError(f"Layer configured with num_kernels={self.num_kernels}, "
                             f"but Keras kernel implies num_kernels={loaded_num_kernels}.")
        
        if self.input_channels is not None and self.input_channels != loaded_input_channels:
            print(f"Warning: Layer {self.name} was pre-configured with input_channels={self.input_channels}, "
                  f"Keras kernel implies input_channels={loaded_input_channels}. Using value from Keras kernel.")
        
        self.input_channels = loaded_input_channels

        if self.kernel_size != (loaded_kh, loaded_kw):
            raise ValueError(f"Layer configured with kernel_size={self.kernel_size}, "
                             f"but Keras kernel implies kernel_size=({loaded_kh}, {loaded_kw}).")

        self.kernel = Tensor(kernel_np, tensor_type=f"{self.name}_kernel")
        self.weights = [self.kernel]

        if self.use_bias:
            if len(keras_weights) == 2:
                keras_bias_np = keras_weights[1].astype(float)
                if keras_bias_np.ndim != 1 or keras_bias_np.shape[0] != self.num_kernels:
                    raise ValueError(f"Keras bias should be 1D with shape ({self.num_kernels},), "
                                     f"got {keras_bias_np.shape}.")
                
                self.bias = Tensor(keras_bias_np, tensor_type=f"{self.name}_bias")
            else:
                print(f"Warning: Layer {self.name} uses bias, but Keras weights list did not include bias. Initializing bias to zeros.")
                bias_data_np = np.zeros((self.num_kernels,), dtype=float)
                self.bias = Tensor(bias_data_np, tensor_type=f"{self.name}_bias")

            self.weights.append(self.bias)
        
        elif len(keras_weights) == 2 and self.use_bias is False:
            print(f"Warning: Layer {self.name} has use_bias=False, but Keras weights list included a bias array. It will be ignored.")
            self.bias = None
        
        self._weights_initialized = True


    def forward(self, input_tensor: Tensor) -> Tensor:
        if input_tensor.ndim != 4:
            raise ValueError(
                f"Input to {self.name} must be 4D (N, C_in, H_in, W_in), "
                f"but got {input_tensor.ndim}D with shape {input_tensor.shape}."
            )
        
        N, C_in_from_input, H_in, W_in = input_tensor.shape
        
        self._ensure_weights_initialized(C_in_from_input)
        
        if self.kernel is None:
            raise RuntimeError(f"Kernel for layer {self.name} is not initialized.")

        convolved_output = convolve2d(input_tensor, self.kernel, self.strides, self.padding)

        if self.use_bias and self.bias is not None:
            bias_reshaped_for_add = self.bias.reshape(1, self.num_kernels, 1, 1)
            output_after_bias = convolved_output + bias_reshaped_for_add
        else:
            output_after_bias = convolved_output
        
        if self.activation_fn_class is not None:
            final_output = output_after_bias.compute_activation(self.activation_fn_class)
        else:
            final_output = output_after_bias
            
        self.output = final_output
        return final_output