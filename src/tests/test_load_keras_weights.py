import unittest
import numpy as np
import os
import tempfile 

try:
    import tensorflow as tf
    
    from tensorflow.keras.layers import Conv2D as KerasConv2D 
    from tensorflow.keras.layers import MaxPooling2D as KerasMaxPooling2D
    from tensorflow.keras.layers import AveragePooling2D as KerasAveragePooling2D
    from tensorflow.keras.layers import Flatten as KerasFlatten
    from tensorflow.keras.layers import Dense as KerasDense 
    from tensorflow.keras.layers import Input as KerasInput
    from tensorflow.keras.models import Sequential as KerasSequentialModel 
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print("Warning: TensorFlow/Keras not found. Skipping Keras weight loading tests.")


from ..functions import ReLU, Sigmoid 
from ..layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten 
from ..model import Sequential 


@unittest.skipIf(not KERAS_AVAILABLE, "TensorFlow/Keras is not installed, skipping Keras interop tests.")
class TestLoadKerasWeights(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.h5_filepath = os.path.join(self.temp_dir.name, "test_keras_weights.h5")

    def tearDown(self):
        self.temp_dir.cleanup()

    def assert_weights_loaded_correctly(self, custom_layer, keras_layer_weights_list):
        """
        Compares weights of a custom layer with a list of Keras NumPy weights.
        Handles Conv2D specifics.
        """
        custom_params = custom_layer.get_parameters()

        if isinstance(custom_layer, Conv2D): 
            self.assertEqual(len(custom_params), len(keras_layer_weights_list),
                             f"Param count mismatch for Conv2D layer {custom_layer.name}")
            
            expected_kernel_transposed = np.transpose(keras_layer_weights_list[0], (3, 2, 0, 1))
            np.testing.assert_array_almost_equal(custom_params[0].data, expected_kernel_transposed,
                                                 err_msg=f"Kernel mismatch for {custom_layer.name}")
            if custom_layer.use_bias: 
                self.assertTrue(len(custom_params) > 1, f"Bias Tensor expected in custom layer {custom_layer.name} but not found in get_parameters()")
                self.assertTrue(len(keras_layer_weights_list) > 1, f"Bias weights expected from Keras for {custom_layer.name} but not found.")
                np.testing.assert_array_almost_equal(custom_params[1].data, keras_layer_weights_list[1],
                                                     err_msg=f"Bias mismatch for {custom_layer.name}")
        else:
            if custom_params: 
                self.assertEqual(len(custom_params), len(keras_layer_weights_list),
                                 f"Param count mismatch for layer {custom_layer.name}")
                for i, p_tensor in enumerate(custom_params):
                    np.testing.assert_array_almost_equal(p_tensor.data, keras_layer_weights_list[i],
                                                         err_msg=f"Weight mismatch for param {i} in {custom_layer.name}")


    def test_load_weights_simple_cnn(self):
        print("\n--- TestKerasLoad: Simple CNN ---")
        
        keras_model = KerasSequentialModel([
            KerasInput(shape=(8, 8, 1), name="input_layer_keras"), 
            KerasConv2D(2, (3,3), padding='same', activation='relu', name="conv2d_keras_1", data_format="channels_last"),
            KerasMaxPooling2D((2,2), name="maxpool_keras_1", data_format="channels_last"),
            KerasFlatten(name="flatten_keras_1"),
            
        ])
        dummy_keras_input = np.random.rand(1, 8, 8, 1).astype(np.float32)
        _ = keras_model(dummy_keras_input) 
        keras_model.save_weights(self.h5_filepath)

        custom_conv1 = Conv2D(num_kernels=2, kernel_size=3, input_channels=1, 
                              padding='same', activation=ReLU, name="my_conv1") 
        custom_pool1 = MaxPooling2D(pool_size=2, name="my_pool1")
        custom_flatten = Flatten(name="my_flatten")
        custom_model = Sequential([custom_conv1, custom_pool1, custom_flatten])

        name_map = {
            "conv2d_keras_1": "my_conv1",
            
        }

        custom_model.load_weights_from_keras_h5(self.h5_filepath, custom_layer_name_map=name_map, skip_missing_layers=True)

        keras_conv1_h5_layer = keras_model.get_layer("conv2d_keras_1")
        if keras_conv1_h5_layer.get_weights(): 
             keras_conv1_weights = keras_conv1_h5_layer.get_weights()
             self.assert_weights_loaded_correctly(custom_conv1, keras_conv1_weights)

        print("Simple CNN weight loading test PASSED.")

    def test_load_weights_multiple_conv_and_pooling(self):
        print("\n--- TestKerasLoad: Multi Conv/Pool ---")
        keras_model = KerasSequentialModel([
            KerasInput(shape=(16, 16, 1)),
            KerasConv2D(4, (3,3), padding='same', activation='relu', name="k_conv1", data_format="channels_last"),
            KerasMaxPooling2D((2,2), name="k_pool1", data_format="channels_last"),
            KerasConv2D(8, (3,3), padding='valid', activation='sigmoid', name="k_conv2", data_format="channels_last"),
            KerasAveragePooling2D((2,2), name="k_pool2", data_format="channels_last"),
            KerasFlatten(name="k_flatten")
        ])
        _ = keras_model(np.random.rand(1, 16, 16, 1).astype(np.float32))
        keras_model.save_weights(self.h5_filepath)
        
        c_conv1 = Conv2D(num_kernels=4, kernel_size=3, input_channels=1, padding='same', activation=ReLU, name="c_conv1")
        c_pool1 = MaxPooling2D(pool_size=2, name="c_pool1")
        
        c_conv2 = Conv2D(num_kernels=8, kernel_size=3, padding='valid', activation=Sigmoid, name="c_conv2")
        c_pool2 = AveragePooling2D(pool_size=2, name="c_pool2")
        c_flatten = Flatten(name="c_flatten")

        custom_model = Sequential([c_conv1, c_pool1, c_conv2, c_pool2, c_flatten])

        name_map = {
            "k_conv1": "c_conv1",
            "k_conv2": "c_conv2"
            
        }
        custom_model.load_weights_from_keras_h5(self.h5_filepath, name_map, skip_missing_layers=True)
        
        keras_c1_w = keras_model.get_layer("k_conv1").get_weights()
        self.assert_weights_loaded_correctly(c_conv1, keras_c1_w)
        
        keras_c2_w = keras_model.get_layer("k_conv2").get_weights()
        self.assert_weights_loaded_correctly(c_conv2, keras_c2_w)
        print("Multi Conv/Pool weight loading test PASSED.")


    def test_load_weights_skip_missing_and_mismatched_names(self):
        print("\n--- TestKerasLoad: Skip Missing / Mismatched Names ---")
        keras_model = KerasSequentialModel([
            KerasInput(shape=(8,8,1)),
            KerasConv2D(2, (3,3), name="conv1_k", use_bias=True), 
            KerasConv2D(4, (3,3), name="conv2_k_extra", use_bias=False), 
            KerasFlatten(name="flatten_k")
        ])
        _ = keras_model(np.random.rand(1,8,8,1).astype(np.float32))
        keras_model.save_weights(self.h5_filepath)

        custom_c1 = Conv2D(num_kernels=2, kernel_size=3, input_channels=1, name="my_conv1", use_bias=True)
        custom_f = Flatten(name="my_flatten")
        
        custom_model_partial = Sequential([custom_c1, custom_f])
        name_map = {"conv1_k": "my_conv1"} 
        
        custom_model_partial.load_weights_from_keras_h5(self.h5_filepath, name_map, skip_missing_layers=True)
        keras_c1_weights = keras_model.get_layer("conv1_k").get_weights()
        self.assert_weights_loaded_correctly(custom_c1, keras_c1_weights)
        print("Skip missing (H5 has extra layer, map used) PASSED.")
        
        custom_model_for_error = Sequential([custom_c1]) 
        map_to_non_existent = {"non_existent_keras_layer": "my_conv1"}
        with self.assertRaisesRegex(ValueError, "not found in H5 file"):
            custom_model_for_error.load_weights_from_keras_h5(self.h5_filepath, map_to_non_existent, skip_missing_layers=False)
        print("Error on mapped layer not in H5 (skip=False) PASSED.")

        custom_c_no_match = Conv2D(num_kernels=2, kernel_size=3, input_channels=1, name="conv_not_in_h5")
        model_with_no_match = Sequential([custom_c_no_match])
        with self.assertRaisesRegex(ValueError, "conv_not_in_h5.*not found in H5 file"):
            model_with_no_match.load_weights_from_keras_h5(self.h5_filepath, skip_missing_layers=False)
        print("Error on direct name match not in H5 (skip=False) PASSED.")

        custom_c1_for_skip = Conv2D(num_kernels=2, kernel_size=3, input_channels=1, name="conv1_k") 
        custom_c_unmatched_in_model = Conv2D(num_kernels=8, kernel_size=3, input_channels=4, name="my_conv_not_in_keras")
        
        model_with_extra_custom = Sequential([custom_c1_for_skip, custom_c_unmatched_in_model])
        try:
            model_with_extra_custom.load_weights_from_keras_h5(self.h5_filepath, custom_layer_name_map=None, skip_missing_layers=True)
            self.assert_weights_loaded_correctly(custom_c1_for_skip, keras_model.get_layer("conv1_k").get_weights())
            self.assertIsNone(custom_c_unmatched_in_model.kernel, 
                              "Unmatched layer's kernel should remain None if not auto-initialized.")
        except Exception as e:
            self.fail(f"load_weights with skip_missing_layers=True and extra custom layer failed: {e}")
        print("Skip missing (Custom model has extra layer, skip=True) PASSED.")