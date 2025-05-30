import numpy as np
import tensorflow as tf

class SimpleRNN(object):
    def __init__(self, units: int, return_sequences: bool = False, go_backwards: bool = False, input_size: int = 1, activation: str = 'tanh') -> None:
        self.return_sequences = return_sequences
        self.go_backwards = go_backwards
        self.units = units
        self.input_size = input_size
        self.activation_str = activation
        
        self.U_t = None
        self.W_t = None
        self.b_xh = None
        self.h_0 = np.zeros((units, 1), dtype=np.float32)

        if activation == 'tanh':
            self.activation_function = np.tanh
        else:
            self.activation_function = np.tanh
            if activation is not None:
                 print(f"Peringatan SimpleRNN: aktivasi '{activation}' tidak secara eksplisit didukung, menggunakan tanh.")


    def _initialize_weights(self, input_size_runtime=None):
        current_input_size = self.input_size if self.input_size is not None else input_size_runtime
        if current_input_size is None:
            raise ValueError("input_size untuk SimpleRNN harus ditentukan untuk inisialisasi bobot.")

        if self.U_t is None:
            self.U_t = np.random.randn(self.units, current_input_size).astype(np.float32) * 0.01
        if self.W_t is None:
            self.W_t = np.random.randn(self.units, self.units).astype(np.float32) * 0.01
        if self.b_xh is None:
            self.b_xh = np.zeros((self.units, 1), dtype=np.float32)

    def forward(self, input_sequence: np.ndarray) -> np.ndarray:
        if tf.is_tensor(input_sequence):
            input_sequence = input_sequence.numpy()

        batch_size, num_timesteps, features_dim = input_sequence.shape

        if self.U_t is None or self.W_t is None or self.b_xh is None:
            self._initialize_weights(features_dim)
        
        if features_dim != self.U_t.shape[1]:
            raise ValueError(
                f"Dimensi fitur input ({features_dim}) tidak cocok dengan dimensi input bobot U_t yang diharapkan ({self.U_t.shape[1]}). "
                f"Pastikan input_size lapisan ({self.input_size}) benar."
            )

        loop_indices = range(num_timesteps)
        if self.go_backwards:
            loop_indices = reversed(loop_indices)

        all_h_t_transposed = []
        current_h_state = np.tile(self.h_0, (1, batch_size)).astype(input_sequence.dtype)


        for i in loop_indices:
            input_i_batch = input_sequence[:, i, :].T # (features_dim, batch_size)

            # (units, features_dim) * (features_dim, batch_size) -> (units, batch_size)
            term_input = np.dot(self.U_t, input_i_batch)
            # (units, units) * (units, batch_size) -> (units, batch_size)
            term_hidden = np.dot(self.W_t, current_h_state)

            h_t = self.activation_function(term_input + term_hidden + self.b_xh) # b_xh

            if self.return_sequences:
                all_h_t_transposed.append(h_t)
            
            current_h_state = h_t

        if self.return_sequences:
            output_stacked_transposed = np.stack(all_h_t_transposed, axis=0)
            output = output_stacked_transposed.transpose(2, 0, 1)
            if self.go_backwards: 
                output = output[:, ::-1, :]
        else:
            output = current_h_state.T

        return output.astype(input_sequence.dtype)

class Bidirectional(object):
    def __init__(self, units: int, merge_mode: str = 'concat', return_sequences: bool = False, input_size: int = None, rnn_activation: str = 'tanh'):
        if merge_mode not in ['concat', 'sum', 'mul', 'ave']:
            raise ValueError(f"Mode merge tidak didukung: {merge_mode}. Pilih dari 'concat', 'sum', 'mul', 'ave'.")

        self.units_per_direction = units
        self.merge_mode = merge_mode
        self.return_sequences_bidir = return_sequences
        self.input_size = input_size
        
        self.forward_rnn = SimpleRNN(units=self.units_per_direction, return_sequences=True, go_backwards=False, input_size=self.input_size, activation=rnn_activation)
        self.backward_rnn = SimpleRNN(units=self.units_per_direction, return_sequences=True, go_backwards=True, input_size=self.input_size, activation=rnn_activation)
        
        if self.merge_mode == 'concat':
            self.output_units_bidir = self.units_per_direction * 2
        else:
            self.output_units_bidir = self.units_per_direction

    def forward(self, input_sequence: np.ndarray) -> np.ndarray:
        if tf.is_tensor(input_sequence):
            input_sequence = input_sequence.numpy()
        
        _, _, features_dim = input_sequence.shape
        if self.forward_rnn.input_size is None:
            self.forward_rnn.input_size = features_dim
            self.forward_rnn._initialize_weights()
        if self.backward_rnn.input_size is None:
            self.backward_rnn.input_size = features_dim
            self.backward_rnn._initialize_weights()

        output_fw_seq = self.forward_rnn.forward(input_sequence) # (batch, time, units)
        output_bw_seq = self.backward_rnn.forward(input_sequence) # (batch, time, units)
        
        merged_output_seq = None
        
        if self.merge_mode == 'concat':
            merged_output_seq = np.concatenate((output_fw_seq, output_bw_seq), axis=-1)
        elif self.merge_mode == 'sum':
            merged_output_seq = output_fw_seq + output_bw_seq
        elif self.merge_mode == 'mul':
            merged_output_seq = output_fw_seq * output_bw_seq
        elif self.merge_mode == 'ave':
            merged_output_seq = (output_fw_seq + output_bw_seq) / 2.0
        
        if self.return_sequences_bidir:
            return merged_output_seq.astype(input_sequence.dtype)
        else:
            final_fw_state = output_fw_seq[:, -1, :] 
            final_bw_state = output_bw_seq[:, 0, :] 
            
            if self.merge_mode == 'concat':
                return np.concatenate((final_fw_state, final_bw_state), axis=-1).astype(input_sequence.dtype)
            elif self.merge_mode == 'sum':
                return (final_fw_state + final_bw_state).astype(input_sequence.dtype)
            elif self.merge_mode == 'mul':
                return (final_fw_state * final_bw_state).astype(input_sequence.dtype)
            elif self.merge_mode == 'ave':
                return ((final_fw_state + final_bw_state) / 2.0).astype(input_sequence.dtype)

class Dense(object):
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
                self.act_forward = lambda x: 1 / (1 + np.exp(-x))
            elif self.activation_function_str.lower() == 'softmax':
                self.act_forward = self._softmax
            elif self.activation_function_str.lower() == 'tanh':
                self.act_forward = np.tanh
            else:
                raise ValueError(f"Fungsi aktivasi tidak didukung: {self.activation_function_str}")

    def _initialize_weights_if_needed(self, current_input_size: int):
        if self.W is None:
            if current_input_size is None:
                raise ValueError("input_size untuk Dense harus ditentukan untuk inisialisasi bobot.")
            self.input_size = current_input_size
            self.W = np.random.randn(self.input_size, self.units).astype(np.float32) * 0.01
        if self.use_bias and self.b is None:
            self.b = np.zeros((1, self.units), dtype=np.float32)
        elif not self.use_bias:
            self.b = None

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        if tf.is_tensor(input_data):
            input_data = input_data.numpy()

        current_input_size = input_data.shape[-1]
        if self.W is None or (self.use_bias and self.b is None and self.b is not False):
            self._initialize_weights_if_needed(current_input_size)
        
        if self.input_size != current_input_size:
             raise ValueError(f"Dimensi input data ({current_input_size}) tidak cocok dengan input_size lapisan Dense ({self.input_size}).")

        linear_output = np.matmul(input_data, self.W)
        if self.use_bias and self.b is not None:
            linear_output += self.b 

        if self.act_forward:
            return self.act_forward(linear_output).astype(input_data.dtype if input_data.dtype in [np.float32, np.float64] else np.float32)
        else:
            return linear_output.astype(input_data.dtype if input_data.dtype in [np.float32, np.float64] else np.float32)

class Embedding(object):
    def __init__(self, input_dim: int, output_dim: int, input_length: int = None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_length = input_length
        
        self.embeddings = np.random.rand(self.input_dim, self.output_dim).astype(np.float32) * 0.01

    def forward(self, input_sequence: np.ndarray) -> np.ndarray:
        if tf.is_tensor(input_sequence):
            input_sequence = input_sequence.numpy()
        
        if not isinstance(input_sequence, np.ndarray) or input_sequence.ndim != 2:
            raise ValueError("Input untuk Embedding harus berupa NumPy array 2D (batch_size, sequence_length).")
        if np.max(input_sequence) >= self.input_dim or np.min(input_sequence) < 0:
            raise ValueError(f"Nilai dalam input_sequence berada di luar jangkauan kosakata [0, {self.input_dim-1}]")

        return self.embeddings[input_sequence].astype(np.float32)

class Dropout(object):
    def __init__(self, rate: float):
        if not (0 <= rate < 1):
            raise ValueError("Rate dropout harus berada di antara 0 (inklusif) dan 1 (eksklusif).")
        self.rate = rate
        self.scale = 1.0 / (1.0 - self.rate) if self.rate < 1.0 and self.rate > 0 else 1.0

    def forward(self, input_data: np.ndarray, training: bool = False) -> np.ndarray:
        if tf.is_tensor(input_data):
            input_data = input_data.numpy()
            
        if not training or self.rate == 0:
            return input_data
        
        mask = np.random.binomial(1, 1.0 - self.rate, size=input_data.shape).astype(input_data.dtype)
        return (input_data * mask) * self.scale

class Model(object):
    def __init__(self, layers: list):
        if not isinstance(layers, list) or not all(hasattr(layer, 'forward') for layer in layers):
            raise ValueError("Parameter 'layers' harus berupa list dari objek lapisan yang memiliki metode 'forward'.")
        self.layers = layers
        self.training_mode = False

    def forward(self, input_data: np.ndarray, training: bool = None) -> np.ndarray:
        if tf.is_tensor(input_data):
            input_data = input_data.numpy()
        
        current_training_mode = training if training is not None else self.training_mode
            
        current_output = input_data
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Dropout):
                current_output = layer.forward(current_output, training=current_training_mode)
            else:
                current_output = layer.forward(current_output)
        return current_output

    def load_weights_from_keras_model(self, file_model: str):
        try:
            keras_model_instance = tf.keras.models.load_model(file_model, compile=False)
        except Exception as e:
            print(f"ERROR: Gagal memuat model Keras dari file '{file_model}'. Kesalahan: {e}")
            import traceback
            traceback.print_exc()
            return

        keras_layers_with_weights = []
        for keras_layer in keras_model_instance.layers:
            if isinstance(keras_layer, (tf.keras.layers.Embedding, tf.keras.layers.Bidirectional, tf.keras.layers.SimpleRNN, tf.keras.layers.Dense)):
                keras_layers_with_weights.append(keras_layer)

        custom_layers_with_weights = []
        for custom_layer in self.layers:
             if isinstance(custom_layer, (Embedding, Bidirectional, SimpleRNN, Dense)):
                custom_layers_with_weights.append(custom_layer)

        if len(custom_layers_with_weights) != len(keras_layers_with_weights):
            print(
                f"PERINGATAN: Jumlah lapisan dengan bobot tidak cocok! "
                f"Kustom: {len(custom_layers_with_weights)} ({[type(l).__name__ for l in custom_layers_with_weights]}), "
                f"Keras: {len(keras_layers_with_weights)} ({[type(l).__name__ for l in keras_layers_with_weights]})."
            )
            print("Pemuatan bobot akan dilanjutkan untuk lapisan yang cocok secara sekuensial hingga batas minimum.")

        num_layers_to_process = min(len(custom_layers_with_weights), len(keras_layers_with_weights))

        for i in range(num_layers_to_process):
            custom_layer = custom_layers_with_weights[i]
            keras_layer = keras_layers_with_weights[i]

            try:
                if isinstance(custom_layer, Embedding) and isinstance(keras_layer, tf.keras.layers.Embedding):
                    k_weights = keras_layer.get_weights()
                    if len(k_weights) == 1:
                        if custom_layer.input_dim != k_weights[0].shape[0] or custom_layer.output_dim != k_weights[0].shape[1]:
                            print(f"PERINGATAN Dimensi Embedding: Kustom ({custom_layer.input_dim},{custom_layer.output_dim}), Keras ({k_weights[0].shape[0]},{k_weights[0].shape[1]}) untuk {keras_layer.name}. Bobot tetap dimuat.")
                        custom_layer.embeddings = k_weights[0].astype(np.float32)
                    else:
                        print(f"PERINGATAN: Jumlah bobot tidak terduga untuk Keras Embedding ({keras_layer.name}): {len(k_weights)}. Diharapkan 1.")

                elif isinstance(custom_layer, Bidirectional) and isinstance(keras_layer, tf.keras.layers.Bidirectional):
                    if not (hasattr(keras_layer, 'forward_layer') and isinstance(keras_layer.forward_layer, tf.keras.layers.SimpleRNN) and hasattr(keras_layer, 'backward_layer') and isinstance(keras_layer.backward_layer, tf.keras.layers.SimpleRNN)):
                        print(f"PERINGATAN: Lapisan terbungkus dalam Keras Bidirectional ({keras_layer.name}) bukan SimpleRNN atau struktur tidak sesuai. Melewati.")
                        continue
                    
                    k_fw_rnn_weights = keras_layer.forward_layer.get_weights()
                    if len(k_fw_rnn_weights) == 3:
                        custom_layer.forward_rnn.U_t = k_fw_rnn_weights[0].T.astype(np.float32)
                        custom_layer.forward_rnn.W_t = k_fw_rnn_weights[1].astype(np.float32)
                        custom_layer.forward_rnn.b_xh = k_fw_rnn_weights[2].reshape(-1, 1).astype(np.float32)
                        custom_layer.forward_rnn.input_size = k_fw_rnn_weights[0].shape[0]
                    else:
                        print(f"PERINGATAN: Jumlah bobot tidak terduga untuk forward_layer Keras Bidirectional ({keras_layer.name}): {len(k_fw_rnn_weights)}. Diharapkan 3.")

                    k_bw_rnn_weights = keras_layer.backward_layer.get_weights()
                    if len(k_bw_rnn_weights) == 3:
                        custom_layer.backward_rnn.U_t = k_bw_rnn_weights[0].T.astype(np.float32)
                        custom_layer.backward_rnn.W_t = k_bw_rnn_weights[1].astype(np.float32)
                        custom_layer.backward_rnn.b_xh = k_bw_rnn_weights[2].reshape(-1, 1).astype(np.float32)
                        custom_layer.backward_rnn.input_size = k_bw_rnn_weights[0].shape[0]
                    else:
                        print(f"PERINGATAN: Jumlah bobot tidak terduga untuk backward_layer Keras Bidirectional ({keras_layer.name}): {len(k_bw_rnn_weights)}. Diharapkan 3.")
                    
                    if custom_layer.input_size is None and custom_layer.forward_rnn.input_size is not None:
                         custom_layer.input_size = custom_layer.forward_rnn.input_size


                elif isinstance(custom_layer, SimpleRNN) and isinstance(keras_layer, tf.keras.layers.SimpleRNN):
                    k_weights = keras_layer.get_weights()
                    if len(k_weights) == 3: 
                        custom_layer.U_t = k_weights[0].T.astype(np.float32)
                        custom_layer.W_t = k_weights[1].astype(np.float32)
                        custom_layer.b_xh = k_weights[2].reshape(-1, 1).astype(np.float32)
                        custom_layer.input_size = k_weights[0].shape[0]
                    else:
                        print(f"PERINGATAN: Jumlah bobot tidak terduga untuk Keras SimpleRNN ({keras_layer.name}): {len(k_weights)}. Diharapkan 3.")

                elif isinstance(custom_layer, Dense) and isinstance(keras_layer, tf.keras.layers.Dense):
                    k_weights = keras_layer.get_weights()
                    
                    keras_input_features = k_weights[0].shape[0]
                    if custom_layer.input_size is None:
                        custom_layer.input_size = keras_input_features
                    elif custom_layer.input_size != keras_input_features:
                        print(f"PERINGATAN input_size Dense: Kustom {custom_layer.input_size}, Keras {keras_input_features} untuk {keras_layer.name}. Bobot tetap dimuat, pastikan kompatibel.")
                    
                    if custom_layer.W is None or custom_layer.W.shape != k_weights[0].shape:
                        custom_layer.W = np.zeros_like(k_weights[0], dtype=np.float32)
                    
                    custom_layer.W = k_weights[0].astype(np.float32)

                    if len(k_weights) == 2:
                        if custom_layer.b is None or custom_layer.b.shape != (1, k_weights[1].shape[0]):
                             custom_layer.b = np.zeros((1, k_weights[1].shape[0]), dtype=np.float32)
                        custom_layer.b = k_weights[1].reshape(1, -1).astype(np.float32)
                        custom_layer.use_bias = True
                        print(f"  Bobot Dense (W dan b) untuk '{keras_layer.name}' dimuat.")
                    elif len(k_weights) == 1:
                        custom_layer.b = None
                        custom_layer.use_bias = False
                    else:
                        print(f"PERINGATAN: Jumlah bobot tidak terduga untuk Keras Dense ({keras_layer.name}): {len(k_weights)}. Diharapkan 1 atau 2.")
                
                else:
                    print(f"PERINGATAN PENTING: Tipe lapisan tidak cocok atau tidak ditangani antara Kustom ({type(custom_layer).__name__}) dan Keras ({type(keras_layer).__name__} - {keras_layer.name}). Bobot TIDAK dimuat untuk pasangan ini.")
            
            except Exception as e:
                print(f"ERROR saat memuat bobot untuk custom_layer '{type(custom_layer).__name__}' dari keras_layer '{keras_layer.name}': {e}")
                import traceback
                traceback.print_exc()