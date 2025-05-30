import numpy as np
import tensorflow as tf

class LSTM_c(object):
    def __init__(self, units: int, return_sequences: bool = False, go_backwards: bool = False, input_size: int = 1) -> None:
        self.return_sequences = return_sequences
        self.go_backwards = go_backwards
        self.units = units
        self.input_size = input_size

        self.U_f = None
        self.W_f = None
        self.b_f = None
        self.U_i = None
        self.W_i = None
        self.b_i = None
        self.U_c = None
        self.W_c = None
        self.b_c = None
        self.U_o = None
        self.W_o = None
        self.b_o = None
        self.h_0 = np.zeros((1, units), dtype=np.float32)

    def _initialize_weights(self, input_size_runtime=None):
        current_input_size = self.input_size if self.input_size is not None else input_size_runtime
        if current_input_size is None:
            raise ValueError("input_size for SimpleRNN must be specified for weight initialization.")

        if self.U_f is None:
            self.U_f = np.random.randn(self.units, current_input_size).astype(np.float32) * 0.01
        if self.W_f is None:
            self.W_f = np.random.randn(self.units, self.units).astype(np.float32) * 0.01
        if self.b_f is None:
            self.b_f = np.zeros((self.units,), dtype=np.float32)
        if self.U_i is None:
            self.U_i = np.random.randn(self.units, current_input_size).astype(np.float32) * 0.01
        if self.W_i is None:
            self.W_i = np.random.randn(self.units, self.units).astype(np.float32) * 0.01
        if self.b_i is None:
            self.b_i = np.zeros((self.units,), dtype=np.float32)
        if self.U_c is None:
            self.U_c = np.random.randn(self.units, current_input_size).astype(np.float32) * 0.01
        if self.W_c is None:
            self.W_c = np.random.randn(self.units, self.units).astype(np.float32) * 0.01
        if self.b_c is None:
            self.b_c = np.zeros((self.units,), dtype=np.float32)
        if self.U_o is None:
            self.U_o = np.random.randn(self.units, current_input_size).astype(np.float32) * 0.01
        if self.W_o is None:
            self.W_o = np.random.randn(self.units, self.units).astype(np.float32) * 0.01
        if self.b_o is None:
            self.b_o = np.zeros((self.units,), dtype=np.float32)

    def forward(self, input_sequence: np.ndarray) -> np.ndarray:
        if tf.is_tensor(input_sequence):
            input_sequence = input_sequence.numpy()

        batch_size, num_timesteps, features_dim = input_sequence.shape

        if self.U_f is None or self.W_f is None or self.b_f is None or self.U_i is None or self.W_i is None or self.b_i is None or self.U_c is None or self.W_c is None or self.b_c is None or self.U_o is None or self.W_o is None or self.b_o is None:
            self._initialize_weights(features_dim)

        if features_dim != self.U_f.shape[1]:
            raise ValueError(
                f"Input feature dimension ({features_dim}) doesn't match expected U_t weight input dimension ({self.U_t.shape[1]}). "
                f"Make sure layer input_size ({self.input_size}) is correct."
            )

        loop_indices = range(num_timesteps)
        if self.go_backwards:
            loop_indices = reversed(loop_indices)

        all_h_t = []
        current_h_state = np.tile(self.h_0, (batch_size, 1)).astype(input_sequence.dtype)

        for i in loop_indices:
            input_i_batch = input_sequence[:, i, :]

            f_t = self._sigmoid(self.U_f @ input_i_batch + self.W_f @ current_h_state + self.b_f)
            i_t = self._sigmoid(self.U_i @ input_i_batch + self.W_i @ current_h_state + self.b_i)
            c_hat_t = self._tanh(self.U_c @ input_i_batch + self.W_c @ current_h_state + self.b_c)
            c_t = f_t * c_t + i_t * c_hat_t
            o_t = self._sigmoid(self.U_o @ input_i_batch + self.W_o @ h_t + self.b_o)
            h_t = o_t * self._tanh(c_t)

            if self.return_sequences:
                all_h_t.append(h_t)

            current_h_state = h_t

        if self.return_sequences:
            output = np.stack(all_h_t, axis=1)
            if self.go_backwards:
                output = output[:, ::-1, :]
        else:
            output = current_h_state

        return output.astype(input_sequence.dtype)
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
    
    def _tanh(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

class Bidirectional_c(object):
    def __init__(self, lstm_layer: LSTM_c, merge_mode: str = 'concat'):
        if merge_mode not in ['concat', 'sum', 'mul', 'ave']:
            raise ValueError(f"Unsupported merge mode: {merge_mode}. Choose from 'concat', 'sum', 'mul', 'ave'.")

        self.merge_mode = merge_mode
        self.return_sequences = lstm_layer.return_sequences  # user's intent

        # Internally we force return_sequences=True so we can slice manually if needed
        import copy
        self.forward_lstm = copy.deepcopy(lstm_layer)
        self.forward_lstm.go_backwards = False
        self.forward_lstm.return_sequences = True

        self.backward_lstm = copy.deepcopy(lstm_layer)
        self.backward_lstm.go_backwards = True
        self.backward_lstm.return_sequences = True

        self.output_units_bidir = (
            lstm_layer.units * 2 if merge_mode == 'concat' else lstm_layer.units
        )

    def forward(self, input_sequence: np.ndarray) -> np.ndarray:
        if tf.is_tensor(input_sequence):
            input_sequence = input_sequence.numpy()

        fw_output = self.forward_lstm.forward(input_sequence)
        bw_output = self.backward_lstm.forward(input_sequence)

        if self.merge_mode == 'concat':
            merged = np.concatenate((fw_output, bw_output), axis=-1)
        elif self.merge_mode == 'sum':
            merged = fw_output + bw_output
        elif self.merge_mode == 'mul':
            merged = fw_output * bw_output
        elif self.merge_mode == 'ave':
            merged = (fw_output + bw_output) / 2.0

        if self.return_sequences:
            return merged.astype(input_sequence.dtype)
        else:
            # Merge final step only
            fw_final = fw_output[:, -1, :]  # last step
            bw_final = bw_output[:, 0, :]   # first step (from reversed)

            if self.merge_mode == 'concat':
                return np.concatenate((fw_final, bw_final), axis=-1).astype(input_sequence.dtype)
            elif self.merge_mode == 'sum':
                return (fw_final + bw_final).astype(input_sequence.dtype)
            elif self.merge_mode == 'mul':
                return (fw_final * bw_final).astype(input_sequence.dtype)
            elif self.merge_mode == 'ave':
                return ((fw_final + bw_final) / 2.0).astype(input_sequence.dtype)

class Dense_c(object):
    def __init__(self, units: int, input_size: int = None, activation_function: str = None, use_bias: bool = True):
        self.units = units
        self.input_size = input_size 
        self.activation_function_str = activation_function
        self.use_bias = use_bias

        self.W = None
        self.b = None

        self._setup_activation()

        if self.input_size is not None:
            self._initialize_weights_if_needed(self.input_size)

    def _setup_activation(self):
        self.act_forward = None
        if self.activation_function_str:
            if self.activation_function_str.lower() == 'relu':
                self.act_forward = lambda x: np.maximum(0, x)
            elif self.activation_function_str.lower() == 'sigmoid':
                self.act_forward = lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow
            elif self.activation_function_str.lower() == 'softmax':
                self.act_forward = self._softmax
            elif self.activation_function_str.lower() == 'tanh':
                self.act_forward = np.tanh
            elif self.activation_function_str.lower() == 'linear':
                self.act_forward = None
            else:
                raise ValueError(f"Unsupported activation function: {self.activation_function_str}")

    def _initialize_weights_if_needed(self, current_input_size: int):
        if self.W is None:
            if current_input_size is None:
                raise ValueError("input_size for Dense_c must be specified for weight initialization.")
            self.input_size = current_input_size
            self.W = np.random.randn(self.input_size, self.units).astype(np.float32) * 0.01
        if self.use_bias and self.b is None:
            self.b = np.zeros((self.units,), dtype=np.float32)  # Fixed: consistent shape
        elif not self.use_bias:
            self.b = None

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        # Clip to prevent overflow
        x_clipped = np.clip(x, -500, 500)
        e_x = np.exp(x_clipped - np.max(x_clipped, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        if tf.is_tensor(input_data):
            input_data = input_data.numpy()

        current_input_size = input_data.shape[-1]
        if self.W is None or (self.use_bias and self.b is None):
            self._initialize_weights_if_needed(current_input_size)

        if self.input_size != current_input_size:
             raise ValueError(f"Input data dimension ({current_input_size}) doesn't match Dense_c layer input_size ({self.input_size}).")

        linear_output = np.matmul(input_data, self.W)
        if self.use_bias and self.b is not None:
            linear_output += self.b

        if self.act_forward:
            return self.act_forward(linear_output).astype(input_data.dtype if input_data.dtype in [np.float32, np.float64] else np.float32)
        else:
            return linear_output.astype(input_data.dtype if input_data.dtype in [np.float32, np.float64] else np.float32)

class Embedding_c(object):
    def __init__(self, input_dim: int, output_dim: int, input_length: int = None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_length = input_length
        
        self.embeddings = np.random.rand(self.input_dim, self.output_dim).astype(np.float32) * 0.01

    def forward(self, input_sequence: np.ndarray) -> np.ndarray:
        if tf.is_tensor(input_sequence):
            input_sequence = input_sequence.numpy()
        
        if not isinstance(input_sequence, np.ndarray) or input_sequence.ndim != 2:
            raise ValueError("Input for Embedding_c must be a 2D NumPy array (batch_size, sequence_length).")
        if np.max(input_sequence) >= self.input_dim or np.min(input_sequence) < 0:
            raise ValueError(f"Values in input_sequence are out of vocabulary range [0, {self.input_dim-1}]")

        return self.embeddings[input_sequence].astype(np.float32)

class Dropout_c(object):
    def __init__(self, rate: float):
        if not (0 <= rate < 1):
            raise ValueError("Dropout_c rate must be between 0 (inclusive) and 1 (exclusive).")
        self.rate = rate
        self.scale = 1.0 / (1.0 - self.rate) if self.rate < 1.0 and self.rate > 0 else 1.0

    def forward(self, input_data: np.ndarray, training: bool = False) -> np.ndarray:
        if tf.is_tensor(input_data):
            input_data = input_data.numpy()
            
        if not training or self.rate == 0:
            return input_data
        
        mask = np.random.binomial(1, 1.0 - self.rate, size=input_data.shape).astype(input_data.dtype)
        return (input_data * mask) * self.scale

class Model_c(object):
    def __init__(self, layers: list):
        if not isinstance(layers, list) or not all(hasattr(layer, 'forward') for layer in layers):
            raise ValueError("Parameter 'layers' must be a list of layer objects that have a 'forward' method.")
        self.layers = layers
        self.training_mode = False

    def predict(self, input_data: np.ndarray, training: bool = None) -> np.ndarray:
        if tf.is_tensor(input_data):
            input_data = input_data.numpy()
        
        current_training_mode = training if training is not None else self.training_mode
            
        current_output = input_data
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Dropout_c):
                current_output = layer.forward(current_output, training=current_training_mode)
            else:
                current_output = layer.forward(current_output)
        return current_output

    def load_weights_from_keras_model(self, keras_model_or_path):
        """Load weights from a Keras model (either model object or file path)"""
        try:
            if isinstance(keras_model_or_path, str):
                keras_model_instance = tf.keras.models.load_model(keras_model_or_path, compile=False)
            else:
                keras_model_instance = keras_model_or_path
        except Exception as e:
            print(f"ERROR: Failed to load Keras model. Error: {e}")
            import traceback
            traceback.print_exc()
            return

        keras_layers_with_weights = []
        for keras_layer in keras_model_instance.layers:
            if isinstance(keras_layer, (tf.keras.layers.Embedding, tf.keras.layers.Bidirectional, tf.keras.layers.LSTM, tf.keras.layers.Dense)):
                keras_layers_with_weights.append(keras_layer)

        custom_layers_with_weights = []
        for custom_layer in self.layers:
             if isinstance(custom_layer, (Embedding_c, Bidirectional_c, LSTM_c, Dense_c)):
                custom_layers_with_weights.append(custom_layer)

        if len(custom_layers_with_weights) != len(keras_layers_with_weights):
            print(
                f"WARNING: Number of layers with weights don't match! "
                f"Custom: {len(custom_layers_with_weights)} ({[type(l).__name__ for l in custom_layers_with_weights]}), "
                f"Keras: {len(keras_layers_with_weights)} ({[type(l).__name__ for l in keras_layers_with_weights]})."
            )

        num_layers_to_process = min(len(custom_layers_with_weights), len(keras_layers_with_weights))

        for i in range(num_layers_to_process):
            custom_layer = custom_layers_with_weights[i]
            keras_layer = keras_layers_with_weights[i]

            try:
                if isinstance(custom_layer, Embedding_c) and isinstance(keras_layer, tf.keras.layers.Embedding):
                    k_weights = keras_layer.get_weights()
                    if len(k_weights) == 1:
                        custom_layer.embeddings = k_weights[0].astype(np.float32)

                elif isinstance(custom_layer, Bidirectional_c) and isinstance(keras_layer, tf.keras.layers.Bidirectional):
                    if not (hasattr(keras_layer, 'forward_layer') and isinstance(keras_layer.forward_layer, tf.keras.layers.SimpleRNN)):
                        print(f"WARNING: Wrapped layer in Keras Bidirectional is not LSTM. Skipping.")
                        continue
                    
                    # Forward RNN
                    k_fw_rnn_weights = keras_layer.forward_layer.get_weights()
                    if len(k_fw_rnn_weights) == 3:
                        custom_layer.forward_rnn.U_t = k_fw_rnn_weights[0].T.astype(np.float32)
                        custom_layer.forward_rnn.W_t = k_fw_rnn_weights[1].astype(np.float32)
                        custom_layer.forward_rnn.b_xh = k_fw_rnn_weights[2].astype(np.float32)
                        custom_layer.forward_rnn.input_size = k_fw_rnn_weights[0].shape[0]

                    # Backward RNN
                    k_bw_rnn_weights = keras_layer.backward_layer.get_weights()
                    if len(k_bw_rnn_weights) == 3:
                        custom_layer.backward_rnn.U_t = k_bw_rnn_weights[0].T.astype(np.float32)
                        custom_layer.backward_rnn.W_t = k_bw_rnn_weights[1].astype(np.float32)
                        custom_layer.backward_rnn.b_xh = k_bw_rnn_weights[2].astype(np.float32)
                        custom_layer.backward_rnn.input_size = k_bw_rnn_weights[0].shape[0]

                    custom_layer.input_size = custom_layer.forward_rnn.input_size

                elif isinstance(custom_layer, LSTM_c) and isinstance(keras_layer, tf.keras.layers.LSTM):
                    k_weights = keras_layer.get_weights()
                    if len(k_weights) == 3: 
                        custom_layer.U_t = k_weights[0].T.astype(np.float32)
                        custom_layer.W_t = k_weights[1].astype(np.float32)
                        custom_layer.b_xh = k_weights[2].astype(np.float32)
                        custom_layer.input_size = k_weights[0].shape[0]

                elif isinstance(custom_layer, Dense_c) and isinstance(keras_layer, tf.keras.layers.Dense):
                    k_weights = keras_layer.get_weights()
                    
                    keras_input_features = k_weights[0].shape[0]
                    custom_layer.input_size = keras_input_features
                    custom_layer.W = k_weights[0].astype(np.float32)

                    if len(k_weights) == 2:
                        custom_layer.b = k_weights[1].astype(np.float32)  # Fixed: consistent shape
                        custom_layer.use_bias = True
                    elif len(k_weights) == 1:
                        custom_layer.b = None
                        custom_layer.use_bias = False
                
                else:
                    print(f"WARNING: Layer type mismatch between Custom ({type(custom_layer).__name__}) and Keras ({type(keras_layer).__name__}). Weights NOT loaded.")
            
            except Exception as e:
                print(f"ERROR loading weights for custom_layer '{type(custom_layer).__name__}' from keras_layer '{keras_layer.name}': {e}")
                import traceback
                traceback.print_exc()