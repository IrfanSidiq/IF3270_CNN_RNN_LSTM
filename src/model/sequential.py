import numpy as np
import time
import math
import h5py

from typing import List, Optional, Tuple, Dict, Union
import inspect

from src.core import Tensor, Layer
from src.functions import LossFunction, MeanSquaredError, CategoricalCrossEntropy, BinaryCrossEntropy
from src.optimization import Optimizer, Adam


class Sequential:
    def __init__(self, layers: Optional[List[Layer]] = None):
        """
        Initializes a Sequential model with a given list of Layer objects.
        """
        self.layers: List[Layer] = []
        if layers:
            for layer in layers:
                self.add(layer)

        self.optimizer: Optional[Optimizer] = None
        self.loss_fn_class: Optional[type[LossFunction]] = None
        
        self._is_compiled: bool = False
        self.history: Dict[str, List[float]] = {}
        self.latest_output_tensor: Optional[Tensor] = None

    def add(self, layer: Layer) -> None:
        """
        Adds a layer to the model.
        """
        if not isinstance(layer, Layer):
            raise TypeError(f"Expected Layer instance, but got {type(layer).__name__}")
        self.layers.append(layer)

    def forward(self, input_tensor: Tensor, training: bool = True) -> Tensor:
        """
        Performs a forward pass through all layers in the model.
        """
        if not self.layers:
            raise ValueError("Cannot perform forward pass on an empty model.")

        x = input_tensor
        for layer in self.layers:
            sig = inspect.signature(layer.forward)
            if 'training' in sig.parameters:
                x = layer.forward(x, training=training)
            else:
                x = layer.forward(x)
        
        self.latest_output_tensor = x
        return x

    def __call__(self, input_tensor: Tensor, training: bool = True) -> Tensor:
        """
        Allows calling the model instance like a function.
        """
        return self.forward(input_tensor, training=training)

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

    def compile(self, optimizer: Optimizer, loss: type[LossFunction]): 
        """
        Configures the model for training.
        """
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"Optimizer must be an Optimizer instance, got {type(optimizer)}.")
        if not (isinstance(loss, type) and issubclass(loss, LossFunction)):
            raise TypeError(f"Loss must be a LossFunction class (e.g., MeanSquaredError), got {type(loss)}.")

        self.optimizer = optimizer
        self.loss_fn_class = loss 

        self.optimizer.set_parameters(self.get_parameters())
        self._is_compiled = True
        print(f"Model compiled with optimizer {self.optimizer.__class__.__name__} and loss {self.loss_fn_class.__name__}.")


    def _training_step(self, x_batch_tensor: Tensor, y_batch_np: np.ndarray) -> float:
        """Performs a single training step: forward, loss, backward, optimizer step."""
        self.optimizer.zero_grad()
        
        y_pred_tensor = self.forward(x_batch_tensor, training=True) 
        current_loss_tensor = self.latest_output_tensor.compute_loss(y_batch_np, self.loss_fn_class)
        batch_loss_value = current_loss_tensor.data[0] 
        
        current_loss_tensor.backward() 
        
        self.optimizer.step()
        return batch_loss_value


    def fit(self, 
            x_train: np.ndarray, 
            y_train: np.ndarray, 
            epochs: int = 10, 
            batch_size: int = 32,
            validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            shuffle: bool = True,
            verbose: int = 1) -> Dict[str, List[float]]:
        """
        Trains the model for a fixed number of epochs.
        """

        if not self._is_compiled:
            raise RuntimeError("Model must be compiled before training. Call model.compile(...)")
        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError("x_train and y_train must have the same number of samples.")

        num_samples = x_train.shape[0]
        self.history = {'loss': [], 'val_loss': []}

        for epoch in range(epochs):
            epoch_start_time = time.time()
            if verbose >= 1:
                print(f"Epoch {epoch+1}/{epochs}")

            if shuffle:
                indices = np.arange(num_samples)
                np.random.shuffle(indices)
                x_train_shuffled = x_train[indices]
                y_train_shuffled = y_train[indices]
            else:
                x_train_shuffled = x_train
                y_train_shuffled = y_train

            epoch_loss_sum = 0.0
            num_batches = math.ceil(num_samples / batch_size)

            for batch_idx in range(num_batches):
                if verbose == 1:
                    progress = (batch_idx + 1) / num_batches
                    bar_length = 30
                    filled_length = int(bar_length * progress)
                    bar = '█' * filled_length + '-' * (bar_length - filled_length)
                    
                    progress_str = f"\r  {batch_idx+1}/{num_batches} [{bar}] {progress*100:.1f}%"
                    print(progress_str, end="")

                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_samples)
                
                x_batch_np = x_train_shuffled[start_idx:end_idx]
                y_batch_np = y_train_shuffled[start_idx:end_idx]
                
                batch_loss_value = self._training_step(Tensor(x_batch_np), y_batch_np)
                epoch_loss_sum += batch_loss_value * (end_idx - start_idx) 

                if verbose == 1 and batch_idx == num_batches - 1: 
                     print("\r" + " " * (len(progress_str) + 20), end="\r")


            avg_epoch_loss = epoch_loss_sum / num_samples
            self.history['loss'].append(avg_epoch_loss)
            
            log_message = f"  loss: {avg_epoch_loss:.4f}"

            if validation_data:
                x_val, y_val = validation_data
                
                val_loss = self.evaluate(x_val, y_val, batch_size=batch_size, verbose=0) 
                self.history['val_loss'].append(val_loss)
                log_message += f" - val_loss: {val_loss:.4f}"
            else:
                self.history['val_loss'].append(np.nan)

            epoch_duration = time.time() - epoch_start_time
            log_message += f" - {epoch_duration:.2f}s/epoch"
            if verbose >= 1:
                print(log_message)
                
        return self.history

    def predict(self, x_test: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """
        Generates output predictions for the input samples.
        """
        if not self.layers:
            raise ValueError("Model is empty and cannot make predictions.")
        
        num_samples = x_test.shape[0]
        num_batches = math.ceil(num_samples / batch_size)
        all_predictions_list = []

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            x_batch_np = x_test[start_idx:end_idx]
            
            x_batch_tensor = Tensor(x_batch_np)
            y_pred_tensor = self.forward(x_batch_tensor)
            all_predictions_list.append(y_pred_tensor.data)
            
        return np.concatenate(all_predictions_list, axis=0)

    def evaluate(self, 
                 x_test: np.ndarray, 
                 y_test: np.ndarray, 
                 batch_size: int = 32,
                 verbose: int = 1) -> float: 
        """
        Returns the loss value for the model in test mode.
        """
        if not self._is_compiled:
            raise RuntimeError("Model must be compiled with a loss function before calling evaluate.")

        num_samples = x_test.shape[0]
        num_batches = math.ceil(num_samples / batch_size)
        total_loss = 0.0
        
        if verbose == 1: print("Evaluating...")

        for batch_idx in range(num_batches):
            if verbose == 1: 
                progress = (batch_idx + 1) / num_batches
                bar_length = 30
                filled_length = int(bar_length * progress)
                bar = '█' * filled_length + '-' * (bar_length - filled_length)
                print(f"\r  {batch_idx+1}/{num_batches} [{bar}] {progress*100:.1f}%", end="")

            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            x_batch_np = x_test[start_idx:end_idx]
            y_batch_np = y_test[start_idx:end_idx]

            x_batch_tensor = Tensor(x_batch_np)
            y_pred_tensor = self.forward(x_batch_tensor, training=False)
            
            loss_value_np = self.loss_fn_class.forward(y_batch_np, y_pred_tensor.data)
            total_loss += loss_value_np * (end_idx - start_idx) 
        
        if verbose == 1: print() 

        avg_loss = total_loss / num_samples
        if verbose == 1:
            print(f"Evaluation - loss: {avg_loss:.4f}")
        return avg_loss
    
    def load_weights_from_keras_h5(self, filepath: str, 
                                   custom_layer_name_map: Optional[Dict[str, str]] = None,
                                   skip_missing_layers: bool = False):
        """
        Loads weights from a Keras HDF5 file into the model's layers.
        """
        if not self.layers:
            print("Warning: Trying to load weights into an empty Sequential model.")
            return

        print(f"Loading weights from Keras H5 file: {filepath}")
        
        model_layer_name_to_idx: Dict[str, int] = {layer.name: i for i, layer in enumerate(self.layers)}

        with h5py.File(filepath, 'r') as f:
            loaded_keras_layer_names = list(f.keys()) 

            for i, current_custom_layer in enumerate(self.layers):
                custom_layer_name = current_custom_layer.name
                keras_layer_name_to_find = custom_layer_name 

                if custom_layer_name_map and custom_layer_name in custom_layer_name_map:
                    keras_layer_name_to_find = custom_layer_name_map[custom_layer_name]
                elif custom_layer_name_map and keras_layer_name_to_find not in loaded_keras_layer_names:
                    
                    for k_name, c_name in custom_layer_name_map.items():
                        if c_name == custom_layer_name:
                            keras_layer_name_to_find = k_name
                            break
                
                if keras_layer_name_to_find not in f:
                    msg = (f"Layer group '{keras_layer_name_to_find}' (for custom layer '{custom_layer_name}') "
                           f"not found in H5 file. Available groups: {loaded_keras_layer_names}")
                    if skip_missing_layers:
                        print(f"Warning: {msg} Skipping.")
                        continue
                    else:
                        raise ValueError(msg)

                keras_layer_group = f[keras_layer_name_to_find]
                
                weight_datasets = []
                def find_datasets(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        weight_datasets.append(obj)

                keras_layer_group.visititems(find_datasets)

                sorted_keras_weights_np_list = []
                if hasattr(current_custom_layer, 'set_weights_from_keras'): 
                    kernel_ds = None
                    bias_ds = None
                    for ds in weight_datasets:
                        if 'kernel' in ds.name.lower(): kernel_ds = ds
                        elif 'bias' in ds.name.lower(): bias_ds = ds
                    
                    if kernel_ds is not None:
                        sorted_keras_weights_np_list.append(kernel_ds[()]) 
                    else:
                        raise ValueError(f"Kernel not found for layer {keras_layer_name_to_find} in H5 file.")
                    
                    if current_custom_layer.use_bias:
                        if bias_ds is not None:
                            sorted_keras_weights_np_list.append(bias_ds[()])
                        else:
                            print(f"Warning: Bias not found for layer {keras_layer_name_to_find} in H5, "
                                  f"but layer {current_custom_layer.name} uses bias. Will be zero-initialized if needed.")
                    
                    current_custom_layer.set_weights_from_keras(sorted_keras_weights_np_list)
                
                elif not current_custom_layer.get_parameters():
                    print(f"Layer {custom_layer_name} has no loadable weights, skipping.")
                else:
                    msg = (f"Layer {custom_layer_name} (type {type(current_custom_layer).__name__}) "
                           f"does not have a recognized weight loading method "
                           f"('set_weights_from_keras' or 'set_weights_from_keras_bn').")
                    if skip_missing_layers:
                        print(f"Warning: {msg} Skipping.")
                    else:
                        raise NotImplementedError(msg)
            
            print(f"Successfully loaded weights into {i+1} matched layers.")

            if not skip_missing_layers:
                model_keras_names_found = set()
                if custom_layer_name_map:
                    for c_name, k_name in custom_layer_name_map.items():
                        if c_name in model_layer_name_to_idx:
                             model_keras_names_found.add(k_name)
                for layer in self.layers:
                    if custom_layer_name_map and layer.name in custom_layer_name_map.values():
                        continue 
                    if layer.name in loaded_keras_layer_names:
                         model_keras_names_found.add(layer.name)

                unused_keras_layers = set(loaded_keras_layer_names) - model_keras_names_found
                if unused_keras_layers:
                    print(f"Warning: The following Keras layer groups in the H5 file were not loaded "
                          f"into any layer in this Sequential model: {list(unused_keras_layers)}")