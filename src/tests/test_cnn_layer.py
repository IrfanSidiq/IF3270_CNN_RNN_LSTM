import numpy as np
import unittest
import time

from src.core import Tensor
from src.functions import ReLU
from src.utils import convolve2d, _calculate_pooling_output_dims, numerical_gradient_array
from src.layers import Conv2D, Flatten, MaxPooling2D, AveragePooling2D


class TestCNNLayer(unittest.TestCase):
    def assertTensorsAlmostEqual(self, t1: Tensor, t2: Tensor, decimal=6, msg=None):
        np.testing.assert_array_almost_equal(t1.data, t2.data, decimal=decimal, err_msg=msg)

    def assertGradientsAlmostEqual(self, tensor_with_grad: Tensor, expected_grad_np: np.ndarray, decimal=5, msg=None):
        np.testing.assert_array_almost_equal(tensor_with_grad.gradient, expected_grad_np, decimal=decimal, err_msg=msg)

    def run_forward_pass_test_large(self, N, C_in, H_in, W_in, num_kernels, kernel_size,
                                    strides, padding, use_bias, activation_class,
                                    provide_specific_weights=False):
        """Helper for forward pass with potentially specific weights for value checks."""
        print(f"\n--- Forward Test Large: N={N},C_in={C_in},H={H_in},W={W_in},K={num_kernels},KS={kernel_size},S={strides},P='{padding}',Bias={use_bias},Act={activation_class.__name__ if activation_class else 'None'} ---")
        start_time = time.time()

        layer = Conv2D(
            num_kernels=num_kernels,
            kernel_size=kernel_size,
            input_channels=C_in if not provide_specific_weights else None,
            strides=strides,
            padding=padding,
            activation=activation_class,
            use_bias=use_bias
        )

        kernel_data_keras_fmt = None
        bias_data_np = None

        if provide_specific_weights:
            kh_l, kw_l = layer.kernel_size
            kernel_data_keras_fmt = np.random.randn(kh_l, kw_l, C_in, num_kernels).astype(float) * 0.01
            keras_weights = [kernel_data_keras_fmt]
            if use_bias:
                bias_data_np = np.random.randn(num_kernels).astype(float) * 0.01
                keras_weights.append(bias_data_np)
            layer.set_weights_from_keras(keras_weights)
        
        input_np = np.random.randn(N, C_in, H_in, W_in).astype(float)
        input_tensor = Tensor(input_np)

        
        output_tensor = layer.forward(input_tensor)
        self.assertIsNotNone(layer.kernel, "Layer kernel not initialized after forward pass")
    
        ref_conv_output_for_shape = convolve2d(input_tensor, layer.kernel, layer.strides, layer.padding)
        expected_output_shape = ref_conv_output_for_shape.shape
        self.assertEqual(output_tensor.shape, expected_output_shape, "Output shape mismatch")

        if provide_specific_weights:
            expected_conv_val = convolve2d(input_tensor, layer.kernel, layer.strides, layer.padding).data
            expected_bias_added_val = expected_conv_val
            if use_bias and layer.bias is not None:
                bias_broadcast = layer.bias.data.reshape(1, num_kernels, 1, 1)
                expected_bias_added_val = expected_conv_val + bias_broadcast
            
            expected_final_val = expected_bias_added_val
            if activation_class is not None:
                expected_final_val = activation_class.forward(expected_bias_added_val)
            
            np.testing.assert_array_almost_equal(output_tensor.data, expected_final_val, decimal=5)
        
        end_time = time.time()
        print(f"Forward Test Large completed in {end_time - start_time:.4f} seconds.")
        return layer, input_tensor, output_tensor

    def run_gradient_check_for_layer_large(self, N, C_in, H_in, W_in, num_kernels, kernel_size,
                                           strides, padding, use_bias, activation_class, 
                                           eps=1e-4, decimal=3):
        """
        Performs numerical gradient checking for kernel and bias of the Conv2D.
        Uses slightly larger eps and lower decimal for stability with more operations.
        """
        print(f"\n--- Grad Check Large: N={N},C_in={C_in},H={H_in},W={W_in},K={num_kernels},KS={kernel_size},S={strides},P='{padding}',Bias={use_bias},Act={activation_class.__name__ if activation_class else 'None'} ---")
        start_time = time.time()

        layer = Conv2D(
            num_kernels=num_kernels,
            kernel_size=kernel_size,
            input_channels=C_in,
            strides=strides,
            padding=padding,
            activation=activation_class,
            use_bias=use_bias
        )
        
        kh, kw = layer.kernel_size
        initial_kernel_np_keras = np.random.randn(kh, kw, C_in, num_kernels).astype(float) * 0.05
        keras_weights_list = [initial_kernel_np_keras]
        initial_bias_np = None
        if use_bias:
            initial_bias_np = np.random.randn(num_kernels).astype(float) * 0.05
            keras_weights_list.append(initial_bias_np)
        layer.set_weights_from_keras(keras_weights_list)

        input_np = np.random.randn(N, C_in, H_in, W_in).astype(float) * 0.5

        original_bias_tensor_const = None
        if layer.use_bias and layer.bias is not None:
            original_bias_tensor_const = Tensor(layer.bias.data.copy())
            original_bias_tensor_const.requires_grad = False       

        original_kernel_tensor_const = None
        if layer.kernel is not None:
            original_kernel_tensor_const = Tensor(layer.kernel.data.copy())
            original_kernel_tensor_const.requires_grad = False


        # --- Check dL/dKernel ---
        def f_kernel(k_data_keras_fmt):
            k_data_internal_fmt = np.transpose(k_data_keras_fmt, (3, 2, 0, 1))
            current_kernel_tensor = Tensor(k_data_internal_fmt)
            current_input_tensor = Tensor(input_np.copy())
            
            conv_out = convolve2d(current_input_tensor, current_kernel_tensor, layer.strides, layer.padding)
            bias_added_out = conv_out
            if layer.use_bias and original_bias_tensor_const is not None:
                bias_broadcast = original_bias_tensor_const.reshape(1, layer.num_kernels, 1, 1)
                bias_added_out = conv_out + bias_broadcast
            final_out_tensor = bias_added_out
            if layer.activation_fn_class is not None:
                final_out_tensor = bias_added_out.compute_activation(layer.activation_fn_class)
            return np.sum(final_out_tensor.data)

        print("Calculating numerical kernel gradient...")
        num_grad_kernel_keras_fmt = numerical_gradient_array(f_kernel, initial_kernel_np_keras.copy(), 1.0, eps)
        num_grad_kernel_internal_fmt = np.transpose(num_grad_kernel_keras_fmt, (3, 2, 0, 1))
        print("Numerical kernel gradient calculated.")

        if layer.kernel: layer.kernel.gradient.fill(0.0)
        if layer.bias: layer.bias.gradient.fill(0.0)

        print("Calculating analytical kernel gradient...")
        input_for_analytical = Tensor(input_np.copy())
        output_analytical = layer.forward(input_for_analytical)
        loss_tensor = output_analytical.sum()
        loss_tensor.backward()
        print("Analytical kernel gradient calculated.")

        self.assertIsNotNone(layer.kernel, "Kernel is None before gradient assertion.")
        self.assertGradientsAlmostEqual(layer.kernel, num_grad_kernel_internal_fmt, decimal=decimal,
                                        msg="Kernel gradient mismatch (Large Input)")
        print("Kernel gradient check PASSED (Large Input).")

        # --- Check dL/dBias (if use_bias) ---
        if use_bias:
            self.assertIsNotNone(layer.bias, "Bias is None when use_bias=True before gradient check.")
            self.assertIsNotNone(original_kernel_tensor_const, "Original kernel const is None for bias check.")

            def f_bias(b_data):
                current_bias_tensor = Tensor(b_data)
                current_input_tensor = Tensor(input_np.copy())
                
                conv_out = convolve2d(current_input_tensor, original_kernel_tensor_const, 
                                      layer.strides, layer.padding)
                bias_broadcast = current_bias_tensor.reshape(1, layer.num_kernels, 1, 1)
                bias_added_out = conv_out + bias_broadcast
                final_out_tensor = bias_added_out
                if layer.activation_fn_class is not None:
                    final_out_tensor = bias_added_out.compute_activation(layer.activation_fn_class)
                return np.sum(final_out_tensor.data)

            print("Calculating numerical bias gradient...")
            num_grad_bias = numerical_gradient_array(f_bias, initial_bias_np.copy(), 1.0, eps)
            print("Numerical bias gradient calculated.")
            
            self.assertGradientsAlmostEqual(layer.bias, num_grad_bias, decimal=decimal,
                                            msg="Bias gradient mismatch (Large Input)")
            print("Bias gradient check PASSED (Large Input).")
        
        end_time = time.time()
        print(f"Grad Check Large completed in {end_time - start_time:.4f} seconds.")

    # --- Test Cases ---

    def test_forward_pass_moderately_large(self):
        self.run_forward_pass_test_large(
            N=4, C_in=8, H_in=28, W_in=28,
            num_kernels=16, kernel_size=(5,5),
            strides=(1,1), padding='same',
            use_bias=True, activation_class=ReLU,
            provide_specific_weights=True
        )

    def test_forward_pass_large_strides(self):
        self.run_forward_pass_test_large(
            N=2, C_in=3, H_in=32, W_in=32,
            num_kernels=8, kernel_size=(3,3),
            strides=(2,2), padding='valid',
            use_bias=False, activation_class=None,
            provide_specific_weights=True
        )

    def test_gradients_moderately_large_no_act_no_bias(self):
        self.run_gradient_check_for_layer_large(
            N=1, C_in=3, H_in=10, W_in=10,
            num_kernels=2, kernel_size=(3,3),
            strides=(1,1), padding='valid',
            use_bias=False, activation_class=None,
            decimal=3
        )
        
    def test_gradients_moderately_large_with_bias_relu_padding_stride(self):
        self.run_gradient_check_for_layer_large(
            N=2, C_in=2, H_in=8, W_in=8,
            num_kernels=3, kernel_size=(3,3),
            strides=(2,2), padding='same',
            use_bias=True, activation_class=ReLU,
            decimal=2
        )

    def test_maxpool_forward_shape_valid(self):
        layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        input_np = np.random.rand(2, 3, 10, 10)
        input_tensor = Tensor(input_np)
        output_tensor = layer.forward(input_tensor)
        
        H_out, W_out, _, _, _, _ = _calculate_pooling_output_dims(10, 10, 2, 2, 2, 2, 'valid')
        self.assertEqual(output_tensor.shape, (2, 3, H_out, W_out))

    def test_maxpool_forward_shape_same(self):
        layer = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')
        input_np = np.random.rand(1, 1, 7, 7)
        input_tensor = Tensor(input_np)
        output_tensor = layer.forward(input_tensor)
        
        H_out, W_out, _, _, _, _ = _calculate_pooling_output_dims(7, 7, 3, 3, 2, 2, 'same')
        self.assertEqual(output_tensor.shape, (1, 1, H_out, W_out))

    def test_maxpool_forward_values_valid(self):
        layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        input_np = np.array([[[[1, 2, 3, 4],
                               [5, 6, 7, 8],
                               [9, 10, 11, 12],
                               [13, 14, 15, 16]]]], dtype=float)
        input_tensor = Tensor(input_np)
        output_tensor = layer.forward(input_tensor)
        
        expected_output_np = np.array([[[[6, 8],
                                         [14, 16]]]], dtype=float)
        self.assertTensorsAlmostEqual(output_tensor, expected_output_np)

    def test_maxpool_gradient(self):
        print("\n--- Grad Check: MaxPooling2DLayer ---")
        N, C, H, W = 2, 2, 6, 6
        pool_size, strides, padding = (2,2), (2,2), 'valid'
        
        layer = MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding)
        input_np = np.random.randn(N, C, H, W) * 10

        def f_maxpool(x_tensor: Tensor):
            output = layer.forward(x_tensor)
            return np.sum(output.data)

        num_grad = numerical_gradient_array(f_maxpool, input_np.copy(), 1.0, eps=1e-5)

        
        input_tensor_analytical = Tensor(input_np.copy())
        output_analytical = layer.forward(input_tensor_analytical)
        loss_tensor = output_analytical.sum()
        loss_tensor.backward()

        self.assertGradientsAlmostEqual(input_tensor_analytical, num_grad, decimal=4,
                                        msg="MaxPooling gradient mismatch")
        print("MaxPooling gradient check PASSED.")

    def test_maxpool_gradient_stride_less_than_pool_overlapping(self):
        print("\n--- Grad Check: MaxPooling2DLayer (Overlapping) ---")
        N, C, H, W = 1, 1, 4, 4
        pool_size, strides, padding = (3,3), (1,1), 'valid'
        
        layer = MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding)
        input_np = np.arange(N*C*H*W, dtype=float).reshape(N,C,H,W)
        input_np[0,0,1,1] = 1000 

        def f_maxpool(x_tensor: Tensor):
            f_layer = MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding)
            output = f_layer.forward(x_tensor)
            return np.sum(output.data * np.arange(1, output.data.size + 1).reshape(output.shape))

        num_grad = numerical_gradient_array(f_maxpool, input_np.copy(), 1.0, eps=1e-5)

        analytical_layer = MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding)
        input_tensor_analytical = Tensor(input_np.copy())
        output_analytical = analytical_layer.forward(input_tensor_analytical)
        
        weights_np = np.arange(1, output_analytical.data.size + 1).reshape(output_analytical.shape).astype(float)
        weights_tensor = Tensor(weights_np)
        weights_tensor.requires_grad = False

        weighted_output = output_analytical * weights_tensor
        loss_tensor = weighted_output.sum()
        loss_tensor.backward()

        self.assertGradientsAlmostEqual(input_tensor_analytical, num_grad, decimal=4,
                                        msg="MaxPooling overlapping gradient mismatch")
        print("MaxPooling overlapping gradient check PASSED.")


    # --- AveragePooling2DLayer Tests ---
    def test_avgpool_forward_shape_valid(self):
        layer = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        input_np = np.random.rand(2, 3, 10, 10)
        input_tensor = Tensor(input_np)
        output_tensor = layer.forward(input_tensor)
        H_out, W_out, *r = _calculate_pooling_output_dims(10,10,2,2,2,2,'valid')
        self.assertEqual(output_tensor.shape, (2, 3, H_out, W_out))

    def test_avgpool_forward_values_valid(self):
        layer = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        input_np = np.array([[[[1, 2, 3, 4],
                               [5, 6, 7, 8],
                               [9, 10, 11, 12],
                               [13, 14, 15, 16]]]], dtype=float) # 1,1,4,4
        input_tensor = Tensor(input_np)
        output_tensor = layer.forward(input_tensor)
        expected_output_np = np.array([[[[3.5, 5.5],
                                         [11.5, 13.5]]]], dtype=float)
        self.assertTensorsAlmostEqual(output_tensor, expected_output_np)

    def test_avgpool_gradient(self):
        print("\n--- Grad Check: AveragePooling2DLayer ---")
        N, C, H, W = 2, 1, 4, 4
        pool_size, strides, padding = (2,2), (2,2), 'valid'
        
        layer = AveragePooling2D(pool_size=pool_size, strides=strides, padding=padding)
        input_np = np.random.randn(N, C, H, W)

        def f_avgpool(x_tensor: Tensor):
            output = layer.forward(x_tensor)
            return np.sum(output.data)

        num_grad = numerical_gradient_array(f_avgpool, input_np.copy(), 1.0, eps=1e-5)

        input_tensor_analytical = Tensor(input_np.copy())
        output_analytical = layer.forward(input_tensor_analytical)
        loss_tensor = output_analytical.sum()
        loss_tensor.backward()

        self.assertGradientsAlmostEqual(input_tensor_analytical, num_grad, decimal=4,
                                        msg="AveragePooling gradient mismatch")
        print("AveragePooling gradient check PASSED.")

    # --- FlattenLayer Tests ---
    def test_flatten_forward_shape_and_values(self):
        layer = Flatten()
        input_np = np.random.rand(2, 3, 4, 5)
        input_tensor = Tensor(input_np)
        output_tensor = layer.forward(input_tensor)
        
        self.assertEqual(output_tensor.shape, (2, 3 * 4 * 5))
        self.assertTensorsAlmostEqual(output_tensor, input_np.reshape(2, -1))

    def test_flatten_gradient(self):
        print("\n--- Grad Check: FlattenLayer ---")
        N, C, H, W = 2, 3, 2, 2
        
        layer = Flatten()
        input_np = np.random.randn(N, C, H, W)

        def f_flatten(x_tensor: Tensor):
            f_layer = Flatten() 
            output = f_layer.forward(x_tensor)
            return np.sum(output.data * np.arange(1, output.data.size + 1).reshape(output.shape))

        num_grad = numerical_gradient_array(f_flatten, input_np.copy(), 1.0, eps=1e-5)

        analytical_layer = Flatten()
        input_tensor_analytical = Tensor(input_np.copy())
        output_analytical = analytical_layer.forward(input_tensor_analytical)
        
        fixed_weights_np = np.arange(1, output_analytical.data.size + 1).reshape(output_analytical.shape).astype(float)
        fixed_weights_tensor = Tensor(fixed_weights_np)
        fixed_weights_tensor.requires_grad = False 

        product_tensor = output_analytical * fixed_weights_tensor
        loss_tensor = product_tensor.sum()
        loss_tensor.backward()

        self.assertGradientsAlmostEqual(input_tensor_analytical, num_grad, decimal=4,
                                        msg="Flatten gradient mismatch")
        print("Flatten gradient check PASSED.")