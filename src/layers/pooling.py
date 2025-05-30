import numpy as np

from typing import Tuple, Union, Optional

from ..core import Tensor, Layer
from ..utils import _calculate_pooling_output_dims


class MaxPooling2D(Layer):
    def __init__(self,
                 pool_size: Union[int, Tuple[int, int]] = (2, 2),
                 strides: Optional[Union[int, Tuple[int, int]]] = None,
                 padding: str = 'valid',
                 name: Optional[str] = None):
        
        super().__init__()
        self.pool_size = (pool_size, pool_size) if isinstance(pool_size, int) else pool_size
        if len(self.pool_size) != 2:
            raise ValueError("pool_size must be an int or a tuple of 2 ints.")
        
        self.strides = strides if strides is not None else self.pool_size
        if isinstance(self.strides, int):
            self.strides = (self.strides, self.strides)
        if len(self.strides) != 2:
            raise ValueError("strides must be an int or a tuple of 2 ints.")

        if padding.lower() not in ['valid', 'same']:
            raise ValueError("padding must be 'valid' or 'same'.")
        self.padding = padding.lower()
        
        self.name = name if name else f"max_pooling2d_{id(self)}"
        self._input_shape_during_forward: Optional[Tuple[int, ...]] = None
        self._max_indices: Optional[np.ndarray] = None
        self._weights_initialized = True 

    def forward(self, input_tensor: Tensor) -> Tensor:
        if input_tensor.ndim != 4:
            raise ValueError(f"Input to {self.name} must be 4D (N, C, H_in, W_in), "
                             f"got {input_tensor.ndim}D with shape {input_tensor.shape}.")
        
        self._input_shape_during_forward = input_tensor.shape
        x_data = input_tensor.data
        N, C, H_in, W_in = x_data.shape
        pool_h, pool_w = self.pool_size
        stride_h, stride_w = self.strides

        H_out, W_out, pad_t, pad_b, pad_l, pad_r = _calculate_pooling_output_dims(
            H_in, W_in, pool_h, pool_w, stride_h, stride_w, self.padding
        )

        x_padded_data = x_data
        if self.padding == 'same' and (pad_t > 0 or pad_b > 0 or pad_l > 0 or pad_r > 0) :
             
            x_padded_data = np.pad(
                x_data,
                ((0,0), (0,0), (pad_t, pad_b), (pad_l, pad_r)),
                mode='constant', constant_values=-np.inf
            )
        
        output_data = np.zeros((N, C, H_out, W_out), dtype=float)
        self._max_indices = np.zeros((N, C, H_out, W_out, 2), dtype=int) 

        for n in range(N):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * stride_h
                        h_end = h_start + pool_h
                        w_start = w * stride_w
                        w_end = w_start + pool_w
                        
                        receptive_field = x_padded_data[n, c, h_start:h_end, w_start:w_end]
                        output_data[n, c, h, w] = np.max(receptive_field)
                        
                        idx_in_field = np.unravel_index(np.argmax(receptive_field), receptive_field.shape)
                        
                        self._max_indices[n, c, h, w, 0] = h_start + idx_in_field[0]
                        self._max_indices[n, c, h, w, 1] = w_start + idx_in_field[1]
                        
        res_tensor = Tensor(output_data, [input_tensor], self.name)
        res_tensor._padding_info_for_pool = (pad_t, pad_b, pad_l, pad_r) 
        res_tensor._input_padded_shape_for_pool = x_padded_data.shape 

        def __backward():
            if not input_tensor.requires_grad:
                return

            dL_dY = res_tensor.gradient 
            dL_dX_padded = np.zeros(res_tensor._input_padded_shape_for_pool, dtype=float)
            _N, _C, _H_out, _W_out = dL_dY.shape

            for n_idx in range(_N):
                for c_idx in range(_C):
                    for h_idx in range(_H_out):
                        for w_idx in range(_W_out):
                            max_h_idx_in_padded = self._max_indices[n_idx, c_idx, h_idx, w_idx, 0]
                            max_w_idx_in_padded = self._max_indices[n_idx, c_idx, h_idx, w_idx, 1]
                                            
                            dL_dX_padded[n_idx, c_idx, max_h_idx_in_padded, max_w_idx_in_padded] += dL_dY[n_idx, c_idx, h_idx, w_idx]
            
            _pad_t, _pad_b, _pad_l, _pad_r = res_tensor._padding_info_for_pool
            H_in_orig, W_in_orig = self._input_shape_during_forward[2], self._input_shape_during_forward[3]

            if self.padding == 'same' and (_pad_t > 0 or _pad_b > 0 or _pad_l > 0 or _pad_r > 0):
                input_tensor.gradient += dL_dX_padded[:, :, _pad_t : _pad_t + H_in_orig, _pad_l : _pad_l + W_in_orig]
            else: 
                input_tensor.gradient += dL_dX_padded
        
        res_tensor._Tensor__backward = __backward
        self.output = res_tensor
        return res_tensor
    

class AveragePooling2D(Layer):
    def __init__(self,
                 pool_size: Union[int, Tuple[int, int]] = (2, 2),
                 strides: Optional[Union[int, Tuple[int, int]]] = None,
                 padding: str = 'valid', 
                 name: Optional[str] = None):
        super().__init__()
        self.pool_size = (pool_size, pool_size) if isinstance(pool_size, int) else pool_size
        if len(self.pool_size) != 2:
            raise ValueError("pool_size must be an int or a tuple of 2 ints.")

        self.strides = strides if strides is not None else self.pool_size
        if isinstance(self.strides, int):
            self.strides = (self.strides, self.strides)
        if len(self.strides) != 2:
            raise ValueError("strides must be an int or a tuple of 2 ints.")

        if padding.lower() not in ['valid', 'same']:
            raise ValueError("padding must be 'valid' or 'same'.")
        self.padding = padding.lower()
        
        self.name = name if name else f"average_pooling2d_{id(self)}"
        self._input_shape_during_forward: Optional[Tuple[int, ...]] = None
        
        self._weights_initialized = True 

    def forward(self, input_tensor: Tensor) -> Tensor:
        if input_tensor.ndim != 4:
            raise ValueError(f"Input to {self.name} must be 4D (N, C, H_in, W_in), "
                             f"got {input_tensor.ndim}D with shape {input_tensor.shape}.")

        self._input_shape_during_forward = input_tensor.shape
        x_data = input_tensor.data
        N, C, H_in, W_in = x_data.shape
        pool_h, pool_w = self.pool_size
        stride_h, stride_w = self.strides

        H_out, W_out, pad_t, pad_b, pad_l, pad_r = _calculate_pooling_output_dims(
            H_in, W_in, pool_h, pool_w, stride_h, stride_w, self.padding
        )

        x_padded_data = x_data
        if self.padding == 'same' and (pad_t > 0 or pad_b > 0 or pad_l > 0 or pad_r > 0):
            x_padded_data = np.pad(
                x_data,
                ((0,0), (0,0), (pad_t, pad_b), (pad_l, pad_r)),
                mode='constant', constant_values=0.0
            )

        output_data = np.zeros((N, C, H_out, W_out), dtype=float)
        pool_area = pool_h * pool_w

        for n in range(N):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * stride_h
                        h_end = h_start + pool_h
                        w_start = w * stride_w
                        w_end = w_start + pool_w
                        
                        receptive_field = x_padded_data[n, c, h_start:h_end, w_start:w_end]
                        output_data[n, c, h, w] = np.sum(receptive_field) / pool_area
                        
        res_tensor = Tensor(output_data, [input_tensor], self.name)
        res_tensor._padding_info_for_pool = (pad_t, pad_b, pad_l, pad_r)
        res_tensor._input_padded_shape_for_pool = x_padded_data.shape
        res_tensor._pool_size_for_back = self.pool_size 
        res_tensor._strides_for_back = self.strides   

        def __backward():
            if not input_tensor.requires_grad:
                return

            dL_dY = res_tensor.gradient 
            dL_dX_padded = np.zeros(res_tensor._input_padded_shape_for_pool, dtype=float)
            
            _N, _C, _H_out, _W_out = dL_dY.shape
            _pool_h, _pool_w = res_tensor._pool_size_for_back
            _stride_h, _stride_w = res_tensor._strides_for_back
            _pool_area = _pool_h * _pool_w

            for n_idx in range(_N):
                for c_idx in range(_C):
                    for h_idx in range(_H_out):
                        for w_idx in range(_W_out):
                            h_start_bwd = h_idx * _stride_h
                            h_end_bwd = h_start_bwd + _pool_h
                            w_start_bwd = w_idx * _stride_w
                            w_end_bwd = w_start_bwd + _pool_w
                            
                            grad_val_to_distribute = dL_dY[n_idx, c_idx, h_idx, w_idx] / _pool_area
                            dL_dX_padded[n_idx, c_idx, h_start_bwd:h_end_bwd, w_start_bwd:w_end_bwd] += grad_val_to_distribute
            
            _pad_t, _pad_b, _pad_l, _pad_r = res_tensor._padding_info_for_pool
            H_in_orig, W_in_orig = self._input_shape_during_forward[2], self._input_shape_during_forward[3]
            
            if self.padding == 'same' and (_pad_t > 0 or _pad_b > 0 or _pad_l > 0 or _pad_r > 0):
                input_tensor.gradient += dL_dX_padded[:, :, _pad_t : _pad_t + H_in_orig, _pad_l : _pad_l + W_in_orig]
            else: 
                input_tensor.gradient += dL_dX_padded

        res_tensor._Tensor__backward = __backward
        self.output = res_tensor

        return res_tensor