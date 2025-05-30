import unittest
import numpy as np
import time

from src.core import Tensor
from src.layers import Conv2D, MaxPooling2D, Flatten, AveragePooling2D
from src.model import Sequential
from src.functions import ReLU, Sigmoid, MeanSquaredError
from src.optimization import Adam
from src.utils import numerical_gradient_array


class TestCNNModel(unittest.TestCase):
    def assertGradientsAlmostEqual(self, tensor_with_grad: Tensor, expected_grad_np: np.ndarray, 
                                   decimal=3, msg=None): # Default to lower decimal for more complex tests
        if tensor_with_grad is None or tensor_with_grad.gradient is None:
            self.fail(f"Tensor or its gradient is None. Message: {msg}")
        if expected_grad_np is None:
            self.fail(f"Expected gradient NumPy array is None. Message: {msg}")

        self.assertEqual(tensor_with_grad.gradient.shape, expected_grad_np.shape,
                         f"Gradient shape mismatch. Actual: {tensor_with_grad.gradient.shape}, Expected: {expected_grad_np.shape}. {msg}")
        
        np.testing.assert_array_almost_equal(
            tensor_with_grad.gradient, expected_grad_np, decimal=decimal, err_msg=msg
        )

    def test_small_cnn_end_to_end_gradient(self):
        print("\n--- Integration Test: Small CNN End-to-End Gradient Check ---")
        start_time = time.time()

        # --- Model Definition ---
        # Input: (N, C_in, H, W) = (e.g., 1, 1, 8, 8)
        # Conv1: 2 kernels, 3x3, stride 1, padding 'valid', ReLU
        # Pool1: 2x2, stride 2
        # Flatten
        # (No Dense layer yet, so loss will be on flattened output)

        N, C_in, H_in, W_in = 1, 1, 8, 8 # Small input for manageable grad check
        
        # Define initial weights for Conv1 to make gradients non-trivial and reproducible
        # Keras format for kernel: (KH, KW, C_in, num_kernels)
        # Conv1: 2 kernels, C_in=1, KH=3, KW=3
        kh_c1, kw_c1 = 3, 3
        num_k_c1 = 2
        # Use a fixed seed for reproducibility of random weights if needed, or set them manually
        np.random.seed(42) 
        initial_kernel_c1_keras = np.random.randn(kh_c1, kw_c1, C_in, num_k_c1).astype(float) * 0.1
        initial_bias_c1 = np.random.randn(num_k_c1).astype(float) * 0.1
        
        # Create layers that will be used in both numerical and analytical paths
        # We need fresh layer instances for the numerical check's f_model function
        # and for the analytical path to avoid state interference.

        def create_model_instance():
            # This function creates a fresh model instance with fresh layers
            # This is important because layers (like Conv2D) might have internal state
            # or their parameters (kernel/bias Tensors) might be modified.
            conv1 = Conv2D(num_kernels=num_k_c1, kernel_size=(kh_c1, kw_c1), 
                                input_channels=C_in, # Provide for first conv layer
                                padding='valid', activation=ReLU, use_bias=True)
            # Load the pre-defined weights into this fresh conv1 layer
            conv1.set_weights_from_keras([initial_kernel_c1_keras, initial_bias_c1])

            pool1 = MaxPooling2D(pool_size=(2,2), strides=(2,2))
            flatten_layer = Flatten()
            
            model = Sequential([
                conv1,
                pool1,
                flatten_layer
            ])
            return model, conv1 # Return conv1 to access its parameters later

        # --- Numerical Gradient Calculation ---
        input_np = np.random.randn(N, C_in, H_in, W_in).astype(float) * 0.5

        # We need to check gradients for conv1.kernel and conv1.bias
        # The function f_model_param will compute loss by perturbing one specific parameter array
        
        # Helper for f_model_param to set weights of a specific layer in a fresh model
        def set_conv_weights_in_fresh_model(model_layers_list, target_layer_idx, 
                                            kernel_data_keras_fmt, bias_data_np=None):
            conv_layer_to_modify = model_layers_list[target_layer_idx]
            if not isinstance(conv_layer_to_modify, Conv2D):
                raise TypeError("Target layer is not a Conv2D")
            
            weights_to_load = [kernel_data_keras_fmt]
            if conv_layer_to_modify.use_bias and bias_data_np is not None:
                weights_to_load.append(bias_data_np)
            elif conv_layer_to_modify.use_bias and bias_data_np is None: # Default to zeros if bias used but not given
                 weights_to_load.append(np.zeros(conv_layer_to_modify.num_kernels, dtype=float))

            conv_layer_to_modify.set_weights_from_keras(weights_to_load)


        print("Calculating numerical gradients for Conv1 Kernel...")
        def f_model_for_kernel_grad(perturbed_kernel_tensor: Tensor): # Renamed arg for clarity
            # Create a fresh model and its conv1 layer
            temp_model, temp_conv1_layer = create_model_instance() 
            
            # Manually set the perturbed kernel to the fresh conv1 layer
            # Keep its original bias for this kernel perturbation check
            # Pass the .data attribute (NumPy array) to set_weights_from_keras
            temp_conv1_layer.set_weights_from_keras([
                perturbed_kernel_tensor.data, # USE .data HERE
                initial_bias_c1 if temp_conv1_layer.use_bias else None
            ])
            
            output = temp_model.forward(Tensor(input_np.copy())) # Use a copy of input_np
            return np.sum(output.data) # Scalar loss

        # numerical_gradient_array calls f_model_for_kernel_grad with a Tensor,
        # but f_model_for_kernel_grad itself now correctly extracts .data.
        # The first argument to numerical_gradient_array should still be the numpy array to iterate over.
        num_grad_kernel_c1_keras = numerical_gradient_array(f_model_for_kernel_grad, 
                                                            initial_kernel_c1_keras.copy(), # Pass NumPy array
                                                            1.0, eps=1e-4)
        num_grad_kernel_c1_internal = np.transpose(num_grad_kernel_c1_keras, (3, 2, 0, 1))
        print("Numerical gradients for Conv1 Kernel calculated.")

        print("Calculating numerical gradients for Conv1 Bias...")
        def f_model_for_bias_grad(perturbed_bias_tensor: Tensor): # Renamed arg
            temp_model, temp_conv1_layer = create_model_instance()
            # Manually set the perturbed bias, keep original kernel
            temp_conv1_layer.set_weights_from_keras([
                initial_kernel_c1_keras, # This is a NumPy array
                perturbed_bias_tensor.data  # USE .data HERE
            ])
            output = temp_model.forward(Tensor(input_np.copy()))
            return np.sum(output.data)

        num_grad_bias_c1 = numerical_gradient_array(f_model_for_bias_grad, 
                                                    initial_bias_c1.copy(), # Pass NumPy array
                                                    1.0, eps=1e-4)
        print("Numerical gradients for Conv1 Bias calculated.")


        # --- Analytical Gradient Calculation ---
        print("Calculating analytical gradients...")
        analytical_model, analytical_conv1_layer = create_model_instance() # Use a fresh model
        
        # Zero out gradients on the parameters of this specific model instance
        for p in analytical_model.get_parameters():
            p.gradient.fill(0.0)

        input_tensor_analytical = Tensor(input_np.copy())
        output_analytical = analytical_model.forward(input_tensor_analytical)
        
        loss_tensor = output_analytical.sum() # L = sum(output)
        loss_tensor.backward() # Compute all gradients
        print("Analytical gradients calculated.")

        # --- Assertions ---
        # Check Conv1 Kernel Gradient
        self.assertIsNotNone(analytical_conv1_layer.kernel, "Analytical Conv1 kernel is None")
        self.assertGradientsAlmostEqual(analytical_conv1_layer.kernel, 
                                        num_grad_kernel_c1_internal,
                                        msg="Conv1 Kernel gradient mismatch in Sequential model")
        print("Conv1 Kernel gradient check PASSED.")

        # Check Conv1 Bias Gradient
        if analytical_conv1_layer.use_bias: # Only if bias is used
            self.assertIsNotNone(analytical_conv1_layer.bias, "Analytical Conv1 bias is None")
            self.assertGradientsAlmostEqual(analytical_conv1_layer.bias, 
                                            num_grad_bias_c1,
                                            msg="Conv1 Bias gradient mismatch in Sequential model")
            print("Conv1 Bias gradient check PASSED.")
        
        end_time = time.time()
        print(f"Integration Test completed in {end_time - start_time:.4f} seconds.")

    def test_deeper_cnn_end_to_end_gradient(self):
        print("\n--- Integration Test: Deeper CNN End-to-End Gradient Check ---")
        start_time = time.time()

        # --- Model Definition ---
        # Input: (N, C_in, H, W) = (e.g., 1, 1, 14, 14)
        # Conv1: 2 kernels, 3x3, stride 1, padding 'same', ReLU, use_bias=True
        # Pool1: MaxPooling 2x2, stride 2
        # Conv2: 3 kernels, 2x2, stride 1, padding 'valid', Sigmoid, use_bias=True
        # Pool2: AveragePooling 2x2, stride 1 (overlapping)
        # Flatten

        N, C_in, H_in, W_in = 1, 1, 10, 10 # Moderately sized input
        
        # Use a fixed seed for reproducibility
        np.random.seed(123) 

        # --- Initial Weights (Keras format) ---
        # Conv1: 2 kernels, C_in=1, KH=3, KW=3
        kh_c1, kw_c1, num_k_c1 = 3, 3, 2
        initial_kernel_c1_keras = np.random.randn(kh_c1, kw_c1, C_in, num_k_c1).astype(float) * 0.1
        initial_bias_c1 = np.random.randn(num_k_c1).astype(float) * 0.1

        # Conv2: C_in will be num_k_c1=2. 3 kernels, KH=2, KW=2
        kh_c2, kw_c2, num_k_c2 = 2, 2, 3
        initial_kernel_c2_keras = np.random.randn(kh_c2, kw_c2, num_k_c1, num_k_c2).astype(float) * 0.1
        initial_bias_c2 = np.random.randn(num_k_c2).astype(float) * 0.1

        def create_model_instance_with_initial_weights():
            conv1 = Conv2D(num_kernels=num_k_c1, kernel_size=(kh_c1, kw_c1), 
                                input_channels=C_in, padding='same', activation=ReLU, use_bias=True)
            conv1.set_weights_from_keras([initial_kernel_c1_keras, initial_bias_c1])

            pool1 = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="pool1")
            
            conv2 = Conv2D(num_kernels=num_k_c2, kernel_size=(kh_c2, kw_c2), 
                                # input_channels will be inferred from pool1's output (which is num_k_c1)
                                padding='valid', activation=Sigmoid, use_bias=True)
            conv2.set_weights_from_keras([initial_kernel_c2_keras, initial_bias_c2])
            
            pool2 = AveragePooling2D(pool_size=(2,2), strides=(1,1), padding='valid') # Overlapping
            flatten_layer = Flatten(name="flatten")
            
            model = Sequential([conv1, pool1, conv2, pool2, flatten_layer])
            return model, conv1, conv2 # Return layers whose params we check

        # Input data
        input_np = np.random.randn(N, C_in, H_in, W_in).astype(float) * 0.5

        # --- Numerical Gradient Calculation ---
        # We will check gradients for kernel and bias of conv1, and kernel and bias of conv2

        # Function to compute loss given a model and input
        def compute_loss_from_model_output(model_instance: Sequential, input_data_np: np.ndarray):
            output_tensor = model_instance.forward(Tensor(input_data_np))
            # Use a more complex sum to avoid all upstream grads being 1
            # This weights each element of the flattened output differently
            if output_tensor.data.size == 0: return 0.0 # Handle empty output after aggressive pooling
            weights_for_loss = np.arange(1, output_tensor.data.size + 1).astype(float)
            return np.sum(output_tensor.data.flatten() * weights_for_loss)

        print("Calculating numerical gradients for Conv1 Kernel...")
        # f_conv1_kernel now correctly expects a Tensor argument
        def f_conv1_kernel(perturbed_k1_tensor: Tensor): # Argument is a Tensor
            temp_model, temp_c1, _ = create_model_instance_with_initial_weights()
            # Set perturbed kernel for conv1, keep its original bias
            # Extract .data from the Tensor for set_weights_from_keras
            temp_c1.set_weights_from_keras([
                perturbed_k1_tensor.data, # USE .data HERE
                initial_bias_c1 if temp_c1.use_bias else None
            ])
            return compute_loss_from_model_output(temp_model, input_np.copy())
        
        # numerical_gradient_array will pass a Tensor to f_conv1_kernel after perturbing initial_kernel_c1_keras (NumPy)
        num_grad_k1_keras = numerical_gradient_array(f_conv1_kernel, 
                                                     initial_kernel_c1_keras.copy(), # Pass NumPy array to perturb
                                                     1.0, eps=1e-4)
        num_grad_k1_internal = np.transpose(num_grad_k1_keras, (3, 2, 0, 1))
        print("Numerical gradients for Conv1 Kernel calculated.")

        # Check gradients for Conv1 Bias
        print("Calculating numerical gradients for Conv1 Bias...")
        def f_conv1_bias(perturbed_b1_tensor: Tensor): # Argument is a Tensor
            temp_model, temp_c1, _ = create_model_instance_with_initial_weights()
            temp_c1.set_weights_from_keras([
                initial_kernel_c1_keras, # Use original NumPy kernel
                perturbed_b1_tensor.data  # USE .data HERE
            ])
            return compute_loss_from_model_output(temp_model, input_np.copy())
        num_grad_b1 = numerical_gradient_array(f_conv1_bias, 
                                               initial_bias_c1.copy(), # Pass NumPy array to perturb
                                               1.0, eps=1e-4)
        print("Numerical gradients for Conv1 Bias calculated.")

        # Check gradients for Conv2 Kernel
        print("Calculating numerical gradients for Conv2 Kernel...")
        def f_conv2_kernel(perturbed_k2_tensor: Tensor): # Argument is a Tensor
            temp_model, _, temp_c2 = create_model_instance_with_initial_weights()
            temp_c2.set_weights_from_keras([
                perturbed_k2_tensor.data, # USE .data HERE
                initial_bias_c2 if temp_c2.use_bias else None
            ])
            return compute_loss_from_model_output(temp_model, input_np.copy())
        num_grad_k2_keras = numerical_gradient_array(f_conv2_kernel, 
                                                     initial_kernel_c2_keras.copy(), # Pass NumPy array to perturb
                                                     1.0, eps=1e-4)
        num_grad_k2_internal = np.transpose(num_grad_k2_keras, (3, 2, 0, 1))
        print("Numerical gradients for Conv2 Kernel calculated.")

        # Check gradients for Conv2 Bias
        print("Calculating numerical gradients for Conv2 Bias...")
        def f_conv2_bias(perturbed_b2_tensor: Tensor): # Argument is a Tensor
            temp_model, _, temp_c2 = create_model_instance_with_initial_weights()
            temp_c2.set_weights_from_keras([
                initial_kernel_c2_keras, # Use original NumPy kernel
                perturbed_b2_tensor.data  # USE .data HERE
            ])
            return compute_loss_from_model_output(temp_model, input_np.copy())
        num_grad_b2 = numerical_gradient_array(f_conv2_bias, 
                                               initial_bias_c2.copy(), # Pass NumPy array to perturb
                                               1.0, eps=1e-4)
        print("Numerical gradients for Conv2 Bias calculated.")

        # --- Analytical Gradient Calculation ---
        print("Calculating analytical gradients...")
        analytical_model, analytical_c1, analytical_c2 = create_model_instance_with_initial_weights()
        
        for p in analytical_model.get_parameters(): # Zero out all model parameter gradients
            p.gradient.fill(0.0)

        input_tensor_analytical = Tensor(input_np.copy())
        output_analytical = analytical_model.forward(input_tensor_analytical)
        
        # Construct the same weighted loss for analytical pass
        if output_analytical.data.size == 0:
            self.fail("Analytical model output is empty, cannot compute loss.")
        loss_weights_np = np.arange(1, output_analytical.data.size + 1).astype(float)
        loss_weights_tensor = Tensor(loss_weights_np.reshape(output_analytical.data.flatten().shape)) # Match flattened shape
        loss_weights_tensor.requires_grad = False

        # Flatten output_analytical if it's not already (Flatten layer is last)
        # Our FlattenLayer output is already 2D (N, Features)
        # If output_analytical.data is (N, Features), then .flatten() makes it 1D if N=1.
        # Ensure product is element-wise if output_analytical is (N, Features) and loss_weights is (N*Features,)
        
        # If output_analytical is (N, TotalFeatures), its .data.flatten() is (N*TotalFeatures,)
        # loss_weights_tensor should be (TotalFeatures,) if N=1 or broadcastable.
        # The loss_weights_np was created based on output_analytical.data.size, so it's already 1D total size.
        # If output is (N,F) then flatten() -> (N*F). Product with (N*F) loss_weights_tensor.
        
        # If N > 1, output_analytical is (N, Features_flat).
        # To match loss_weights_tensor (1D of size N*Features_flat), we reshape output:
        reshaped_output_for_loss = output_analytical.reshape(output_analytical.data.size) # Reshape to 1D

        product_for_loss = reshaped_output_for_loss * loss_weights_tensor
        loss_tensor = product_for_loss.sum()
        loss_tensor.backward()
        print("Analytical gradients calculated.")

        # --- Assertions ---
        print("Asserting gradients for Conv1 Kernel...")
        self.assertGradientsAlmostEqual(analytical_c1.kernel, num_grad_k1_internal,
                                        msg="Conv1 Kernel gradient mismatch (Deeper CNN)")
        
        print("Asserting gradients for Conv1 Bias...")
        self.assertGradientsAlmostEqual(analytical_c1.bias, num_grad_b1,
                                        msg="Conv1 Bias gradient mismatch (Deeper CNN)")

        print("Asserting gradients for Conv2 Kernel...")
        self.assertGradientsAlmostEqual(analytical_c2.kernel, num_grad_k2_internal,
                                        msg="Conv2 Kernel gradient mismatch (Deeper CNN)")
        
        print("Asserting gradients for Conv2 Bias...")
        self.assertGradientsAlmostEqual(analytical_c2.bias, num_grad_b2,
                                        msg="Conv2 Bias gradient mismatch (Deeper CNN)")
        
        end_time = time.time()
        print(f"Deeper CNN Integration Test completed in {end_time - start_time:.4f} seconds.")
    
    def create_simple_cnn_model(self, input_channels, initial_seed=None) -> Sequential:
        """
        Helper to create a consistent simple CNN model instance.
        """
        if initial_seed is not None:
            np.random.seed(initial_seed)

        # Conv1: 2 kernels, 3x3, input_channels, padding 'same', ReLU, bias
        # Pool1: MaxPooling 2x2
        # Flatten
        # (No final Dense layer yet, output will be flattened features)
        
        # For Conv2DLayer, if weights are not explicitly set, they will be zero-initialized
        # on the first forward pass. For a learning test, we either need to set them
        # randomly or ensure the optimizer can move them from zero.
        # Let's allow them to be zero-initialized for this test to see if Adam can handle it.
        # Alternatively, we can pre-set small random weights.

        conv1 = Conv2D(num_kernels=2, kernel_size=(3,3), 
                            input_channels=input_channels, # Must be provided for the first Conv layer
                            padding='same', activation=ReLU, use_bias=True)
        
        # To make it learnable from zeros, let's add small random weights explicitly
        # This helps avoid issues where all-zero weights + certain activations might lead to zero gradients.
        k1_shape_keras = (3,3, input_channels, 2) # KH, KW, C_in, C_out
        b1_shape = (2,)
        # Re-seed if you want different random weights for different model instances in more complex tests
        if initial_seed is not None: np.random.seed(initial_seed + 10) # Offset seed
        conv1.set_weights_from_keras([
            np.random.randn(*k1_shape_keras) * 0.01,
            np.random.randn(*b1_shape) * 0.01
        ])
        
        pool1 = MaxPooling2D(pool_size=(2,2), strides=(2,2))
        flatten_layer = Flatten(name="flatten_test")
        
        model = Sequential([
            conv1,
            pool1,
            flatten_layer
        ])
        return model

    def test_fit_predict_evaluate_cycle(self):
        print("\n--- TestSequentialModel: Full Fit, Predict, Evaluate Cycle ---")
        
        N_train, C_in, H_in, W_in = 20, 1, 8, 8 # Small dataset for quick test
        N_val = 5

        # 1. Create Model
        model = self.create_simple_cnn_model(input_channels=C_in, initial_seed=42)

        # 2. Prepare Dummy Data
        x_train_np = np.random.rand(N_train, C_in, H_in, W_in).astype(np.float32)
        
        # Determine output shape for y_train by a dry run
        dummy_input_for_shape = Tensor(x_train_np[:1]) # Use one sample
        output_shape_after_flatten = model.forward(dummy_input_for_shape).shape
        self.assertEqual(len(output_shape_after_flatten), 2, "Flattened output should be 2D (N, Features)")
        num_output_features = output_shape_after_flatten[1]

        # Target for y_train: try to make the network output values close to this
        # For simplicity, let y_train be random. The goal is to see if loss decreases.
        y_train_np = np.random.rand(N_train, num_output_features).astype(np.float32) * 2 - 1 # Range [-1, 1]

        x_val_np = np.random.rand(N_val, C_in, H_in, W_in).astype(np.float32)
        y_val_np = np.random.rand(N_val, num_output_features).astype(np.float32) * 2 - 1


        # 3. Compile Model
        optimizer = Adam(learning_rate=0.01) # Higher LR for faster convergence on simple task
        loss_fn_class = MeanSquaredError
        model.compile(optimizer=optimizer, loss=loss_fn_class)
        
        # Store initial parameters (their data) to check if they change after training
        initial_params_data = [p.data.copy() for p in model.get_parameters()]

        # 4. Fit Model
        print("\nStarting model.fit()...")
        epochs = 5 # Short training
        batch_size = 4
        history = model.fit(x_train_np, y_train_np, epochs=epochs, batch_size=batch_size,
                            validation_data=(x_val_np, y_val_np), shuffle=True, verbose=1)
        
        self.assertIn('loss', history)
        self.assertIn('val_loss', history)
        self.assertEqual(len(history['loss']), epochs)
        self.assertEqual(len(history['val_loss']), epochs)

        # Check if loss decreased (it should for a learnable task, even if slightly)
        # This is not a strict guarantee for all random initializations/tasks, but a good sign.
        print(f"Initial loss: {history['loss'][0]}, Final loss: {history['loss'][-1]}")
        self.assertTrue(history['loss'][-1] < history['loss'][0], 
                        "Training loss should decrease over epochs for this learnable task.")

        # Check if parameters actually changed
        final_params_data = [p.data for p in model.get_parameters()]
        params_changed = False
        for initial_p, final_p in zip(initial_params_data, final_params_data):
            if not np.allclose(initial_p, final_p):
                params_changed = True
                break
        self.assertTrue(params_changed, "Model parameters should change after training.")

        # 5. Predict
        print("\nStarting model.predict()...")
        predictions_val = model.predict(x_val_np, batch_size=batch_size)
        self.assertEqual(predictions_val.shape, (N_val, num_output_features), "Predictions shape mismatch.")
        
        predictions_train = model.predict(x_train_np, batch_size=batch_size)
        self.assertEqual(predictions_train.shape, (N_train, num_output_features), "Train predictions shape mismatch.")

        # 6. Evaluate
        print("\nStarting model.evaluate() on validation data...")
        eval_loss_val = model.evaluate(x_val_np, y_val_np, batch_size=batch_size, verbose=1)
        self.assertIsInstance(eval_loss_val, float, "Evaluate should return a float loss.")
        # Check if it's reasonably close to the last validation loss from fit history
        if not np.isnan(history['val_loss'][-1]): # if validation_data was provided to fit
            self.assertAlmostEqual(eval_loss_val, history['val_loss'][-1], places=5,
                                   msg="Evaluate loss should be close to final validation loss from fit.")

        print("\nStarting model.evaluate() on training data...")
        eval_loss_train = model.evaluate(x_train_np, y_train_np, batch_size=batch_size, verbose=0) # Less verbose here
        self.assertIsInstance(eval_loss_train, float)
        self.assertAlmostEqual(eval_loss_train, history['loss'][-1], places=5, # Compare to final training loss
                                   msg="Evaluate loss on train should be close to final training loss from fit.")
        
        print("\n--- TestSequentialModel: Full Cycle Test PASSED ---")


    def test_compile_with_string_optimizers_and_losses_if_supported(self):
        # This test is relevant if your Sequential.compile supports string identifiers
        # Your current simplified Sequential.compile takes Optimizer instance and LossFunction class
        # So, this test would need to be adapted or skipped.
        # For now, let's assume it expects instances/classes as per simplified version.
        
        print("\n--- TestSequentialModel: Compile with specific instances/classes ---")
        model = self.create_simple_cnn_model(input_channels=1)
        
        # Test with Adam instance and MeanSquaredError class
        try:
            model.compile(optimizer=Adam(learning_rate=0.01), loss=MeanSquaredError)
            self.assertIsInstance(model.optimizer, Adam)
            self.assertEqual(model.loss_fn_class, MeanSquaredError)
            print("Compile with Adam instance and MSE class: PASSED")
        except Exception as e:
            self.fail(f"Compile with Adam instance and MSE class failed: {e}")


    def test_get_parameters(self):
        print("\n--- TestSequentialModel: Get Parameters ---")
        model = self.create_simple_cnn_model(input_channels=1, initial_seed=10)
        params = model.get_parameters()
        
        # Our simple model has one Conv2DLayer with kernel and bias
        # Conv1 (kernel + bias) = 2 parameters
        self.assertEqual(len(params), 2, "Should have 2 parameter Tensors (kernel, bias) for the Conv layer.")
        
        conv1 = model.layers[0] # Assuming it's the Conv2DLayer
        self.assertIs(params[0], conv1.kernel, "First parameter should be conv1's kernel.")
        self.assertIs(params[1], conv1.bias, "Second parameter should be conv1's bias.")
        
        for p in params:
            self.assertTrue(p.requires_grad, "All parameters from get_parameters should require_grad.")
        print("Get Parameters test PASSED.")