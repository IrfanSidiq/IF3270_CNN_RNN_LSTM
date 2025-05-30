import numpy as np

from typing import Tuple
from scipy.signal import correlate2d

from src.core import Tensor


def _calculate_padding_2d(input_shape_hw: Tuple[int, int],
                          kernel_shape_hw: Tuple[int, int],
                          strides_hw: Tuple[int, int],
                          padding_mode: str) -> Tuple[int, int, int, int]:
    """
    Calculates padding amounts for 'valid' or 'same'.
    """
    h_in, w_in = input_shape_hw
    kh, kw = kernel_shape_hw
    sh, sw = strides_hw

    if padding_mode == 'valid':
        return (0, 0, 0, 0)
    elif padding_mode == 'same':
        out_h = np.ceil(float(h_in) / float(sh)).astype(int)
        out_w = np.ceil(float(w_in) / float(sw)).astype(int)

        pad_h_total = max(0, (out_h - 1) * sh + kh - h_in)
        pad_w_total = max(0, (out_w - 1) * sw + kw - w_in)

        pad_top = pad_h_total // 2
        pad_bottom = pad_h_total - pad_top
        pad_left = pad_w_total // 2
        pad_right = pad_w_total - pad_left
        return (pad_top, pad_bottom, pad_left, pad_right)
    else:
        raise ValueError(f"Unsupported padding mode: {padding_mode}. Use 'valid' or 'same'.")


def convolve2d(input_tensor: 'Tensor',
                     kernel_tensor: 'Tensor',
                     strides: Tuple[int, int] = (1, 1),
                     padding: str = 'valid') -> 'Tensor':
    """
    Performs 2D convolution.

    Args:
        input_tensor (Tensor): Input data. Shape (N, C_in, H_in, W_in).
        kernel_tensor (Tensor): Kernels. Shape (C_out, C_in, KH, KW).
        strides (Tuple[int, int]): Strides (sh, sw).
        padding (str): 'valid' or 'same'.

    Returns:
        Tensor: Result. Shape (N, C_out, H_out, W_out).
    """
    x_data = input_tensor.data
    k_data = kernel_tensor.data 

    N, C_in_x, H_in, W_in = x_data.shape
    C_out, C_in_k, KH, KW = k_data.shape

    if C_in_x != C_in_k:
        raise ValueError(f"Input channels ({C_in_x}) must match kernel input channels ({C_in_k}).")

    sh, sw = strides

    pad_top, pad_bottom, pad_left, pad_right = _calculate_padding_2d(
        (H_in, W_in), (KH, KW), (1,1), padding 
    )
    
    if padding == 'same':    
        pad_t_manual, pad_b_manual, pad_l_manual, pad_r_manual = _calculate_padding_2d(
            (H_in, W_in), (KH, KW), strides, 'same' 
        )
        x_padded_data_manual = np.pad(
            x_data, 
            ((0,0), (0,0), (pad_t_manual, pad_b_manual), (pad_l_manual, pad_r_manual)), 
            mode='constant', constant_values=0.0
        )
        scipy_padding_mode = 'valid' 
        H_padded_calc, W_padded_calc = x_padded_data_manual.shape[2], x_padded_data_manual.shape[3]
    else: 
        x_padded_data_manual = x_data 
        scipy_padding_mode = 'valid'
        H_padded_calc, W_padded_calc = H_in, W_in
    
    H_out_stride1 = (H_padded_calc - KH) + 1
    W_out_stride1 = (W_padded_calc - KW) + 1
    
    H_out_final = (H_out_stride1 -1) // sh + 1
    W_out_final = (W_out_stride1 -1) // sw + 1

    if H_out_final <= 0 or W_out_final <= 0:
        raise ValueError(f"Output dimensions ({H_out_final}, {W_out_final}) are not positive.")

    output_data = np.zeros((N, C_out, H_out_final, W_out_final), dtype=float)

    for n in range(N):
        for c_o in range(C_out):
            sum_over_input_channels = np.zeros((H_out_stride1, W_out_stride1), dtype=float)
            
            for c_i in range(C_in_x):
                signal = x_padded_data_manual[n, c_i, :, :]
                kernel_slice = k_data[c_o, c_i, :, :]     
                sum_over_input_channels += correlate2d(signal, kernel_slice, mode=scipy_padding_mode)
            
            output_data[n, c_o, :, :] = sum_over_input_channels[::sh, ::sw]


    res_tensor = Tensor(output_data, [input_tensor, kernel_tensor], f"conv2d_scipy(s={strides},p='{padding}')")
    
    res_tensor._padding_mode = padding 
    res_tensor._strides = strides
    res_tensor._x_padded_data_fwd = x_padded_data_manual 
    res_tensor._manual_padding_amounts = (pad_t_manual, pad_b_manual, pad_l_manual, pad_r_manual) if padding == 'same' else (0,0,0,0)

    def __backward():
        dL_dY = res_tensor.gradient 
        x_orig_data = input_tensor.data 
        k_orig_data = kernel_tensor.data 
        
        _N, _C_in, _H_in, _W_in = x_orig_data.shape
        _C_out, _, _KH, _KW = k_orig_data.shape
        s_h, s_w = res_tensor._strides
        
        x_padded_fwd = res_tensor._x_padded_data_fwd 
        H_padded_fwd, W_padded_fwd = x_padded_fwd.shape[2], x_padded_fwd.shape[3]

        dL_dY_dilated_h = (dL_dY.shape[2] - 1) * s_h + 1
        dL_dY_dilated_w = (dL_dY.shape[3] - 1) * s_w + 1
        
        target_dil_h = (H_padded_fwd - _KH) + 1
        target_dil_w = (W_padded_fwd - _KW) + 1

        dL_dY_dilated = np.zeros((_N, _C_out, target_dil_h, target_dil_w), dtype=float)
        dL_dY_dilated[:, :, ::s_h, ::s_w] = dL_dY

        dL_dK = np.zeros_like(k_orig_data)
        if kernel_tensor.requires_grad:
            for n in range(_N):
                for c_o in range(_C_out):
                    for c_i in range(_C_in):
                        signal_for_dk = x_padded_fwd[n, c_i, :, :] 
                        kernel_for_dk = dL_dY_dilated[n, c_o, :, :] 
                        dL_dK[c_o, c_i, :, :] += correlate2d(signal_for_dk, kernel_for_dk, mode='valid')

            kernel_tensor.gradient += dL_dK

        dL_dX_padded_manual = np.zeros_like(x_padded_fwd) 
        if input_tensor.requires_grad:
            k_rot180 = np.rot90(k_orig_data, 2, axes=(2,3)) 

            for n in range(_N):
                for c_i in range(_C_in): 
                    sum_over_output_channels = np.zeros((H_padded_fwd, W_padded_fwd), dtype=float)
                    
                    for c_o in range(_C_out): 
                        signal_for_dx = dL_dY_dilated[n, c_o, :, :] 
                        kernel_for_dx = k_rot180[c_o, c_i, :, :] 
                        sum_over_output_channels += correlate2d(signal_for_dx, kernel_for_dx, mode='full')
                    
                    dL_dX_padded_manual[n, c_i, :, :] = sum_over_output_channels
            
            pad_t_m, pad_b_m, pad_l_m, pad_r_m = res_tensor._manual_padding_amounts
            H_final_unpad_end = dL_dX_padded_manual.shape[2] - pad_b_m
            W_final_unpad_end = dL_dX_padded_manual.shape[3] - pad_r_m
            
            if H_final_unpad_end > pad_t_m and W_final_unpad_end > pad_l_m :
                 input_tensor.gradient += dL_dX_padded_manual[:, :, pad_t_m:H_final_unpad_end, pad_l_m:W_final_unpad_end]
            elif dL_dX_padded_manual.shape[2] == pad_t_m + _H_in and dL_dX_padded_manual.shape[3] == pad_l_m + _W_in : 
                 input_tensor.gradient += dL_dX_padded_manual[:, :, pad_t_m:, pad_l_m:] 
            
    res_tensor._Tensor__backward = __backward
    return res_tensor


def _calculate_pooling_output_dims(input_H, input_W, pool_size_H, pool_size_W, stride_H, stride_W, padding_mode):
    """
    Calculates output dimensions for pooling.
    For 'same' padding, output size = ceil(input_size / stride).
    For 'valid' padding, output size = floor((input_size - pool_size) / stride) + 1.
    """
    if padding_mode.lower() == 'valid':
        H_out = (input_H - pool_size_H) // stride_H + 1
        W_out = (input_W - pool_size_W) // stride_W + 1
        pad_top, pad_bottom, pad_left, pad_right = 0, 0, 0, 0
    elif padding_mode.lower() == 'same':
        H_out = (input_H + stride_H - 1) // stride_H
        W_out = (input_W + stride_W - 1) // stride_W

        needed_input_H = (H_out - 1) * stride_H + pool_size_H
        needed_input_W = (W_out - 1) * stride_W + pool_size_W

        pad_h_total = max(0, needed_input_H - input_H)
        pad_w_total = max(0, needed_input_W - input_W)

        pad_top = pad_h_total // 2
        pad_bottom = pad_h_total - pad_top
        pad_left = pad_w_total // 2
        pad_right = pad_w_total - pad_left
    else:
        raise ValueError("Padding mode must be 'valid' or 'same'.")
    
    if H_out <= 0 or W_out <= 0:
        raise ValueError(f"Output dimensions for pooling ({H_out}, {W_out}) are not positive. "
                         f"Input:({input_H},{input_W}), Pool:({pool_size_H},{pool_size_W}), Stride:({stride_H},{stride_W}), Pad:'{padding_mode}'")

    return H_out, W_out, pad_top, pad_bottom, pad_left, pad_right


def numerical_gradient_array(f, x_numpy_array_original: np.ndarray, df: float, eps: float = 1e-5):
    grad_numerical = np.zeros_like(x_numpy_array_original, dtype=float)
    
    it = np.nditer(x_numpy_array_original, flags=['multi_index'], op_flags=['readonly'])
    
    while not it.finished:
        idx = it.multi_index
        
        x_perturbed_plus = x_numpy_array_original.copy()
        original_val = x_perturbed_plus[idx]
        x_perturbed_plus[idx] = original_val + eps
        fx_plus_eps = f(Tensor(x_perturbed_plus)) 

        x_perturbed_minus = x_numpy_array_original.copy()
        x_perturbed_minus[idx] = original_val - eps
        fx_minus_eps = f(Tensor(x_perturbed_minus))
        
        grad_numerical[idx] = (fx_plus_eps - fx_minus_eps) / (2 * eps)
        it.iternext()
        
    return grad_numerical * df