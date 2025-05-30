import numpy as np
import tensorflow as tf

class SimpleLSTM(object):
    def __init__(self, units: int, return_sequences: bool = False, go_backwards: bool = False, input_size: int = 1) -> None:
        self.return_sequences = return_sequences
        self.go_backwards = go_backwards
        self.units = units
        self.input_size = input_size

        self.W = np.zeros((input_size, 4 * units))
        self.U = np.zeros((units, 4 * units))     
        self.b = np.zeros((4 * units,))

    def forward(self, input_sequence: np.ndarray) -> np.ndarray:
        batch_size, num_timesteps, features_dim = input_sequence.shape
        if features_dim != self.input_size:
            raise ValueError(f"Dimensi fitur input ({features_dim}) tidak cocok dengan input_size lapisan ({self.input_size}).")
        
        loop_indices = range(num_timesteps)
        if self.go_backwards:
            loop_indices = reversed(loop_indices)

        all_h_t_transposed = []
        current_h_state = np.tile(self.h_0, (1, batch_size)).astype(input_sequence.dtype)

        for i in loop_indices:
            input_i_batch = input_sequence[:, i, :].T

            f_t = self._sigmoid(np.dot(self.W, input_i_batch) + np.dot(self.U, current_h_state) + self.b[:self.units])
            i_t = self._sigmoid(np.dot(self.W, input_i_batch) + np.dot(self.U, current_h_state) + self.b[self.units:2*self.units])
            o_t = self._sigmoid(np.dot(self.W, input_i_batch) + np.dot(self.U, current_h_state) + self.b[2*self.units:3*self.units])
            c_tilde = self._tanh(np.dot(self.W, input_i_batch) + np.dot(self.U, current_h_state) + self.b[3*self.units:])
            c_t = f_t * current_h_state + i_t * c_tilde
            h_t = o_t * self._tanh(c_t)

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

    def _tanh(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
    
class BidirectionalLSTM(object):
    def __init__(self, units: int, merge_mode: str = 'concat', return_sequences: bool = False, input_size: int = 1):
        if merge_mode not in ['concat', 'sum', 'mul', 'ave']:
            raise ValueError(f"Mode merge tidak didukung: {merge_mode}. Pilih dari 'concat', 'sum', 'mul', 'ave'.")

        self.units_per_direction = units
        self.merge_mode = merge_mode
        self.return_sequences_bidir = return_sequences
        self.input_size = input_size
        
        self.forward_rnn = SimpleLSTM(units=self.units_per_direction, return_sequences=True, go_backwards=False, input_size=self.input_size)
        self.backward_rnn = SimpleLSTM(units=self.units_per_direction, return_sequences=True, go_backwards=True, input_size=self.input_size)
        
        if self.merge_mode == 'concat':
            self.output_units_bidir = self.units_per_direction * 2
        else:
            self.output_units_bidir = self.units_per_direction

    def forward(self, input_sequence: np.ndarray) -> np.ndarray:
        output_fw_seq = self.forward_rnn.forward(input_sequence)
        output_bw_seq = self.backward_rnn.forward(input_sequence)
        merged_output = None
        if self.merge_mode == 'concat':
            merged_output = np.concatenate((output_fw_seq, output_bw_seq), axis=-1)
        elif self.merge_mode == 'sum':
            merged_output = output_fw_seq + output_bw_seq
        elif self.merge_mode == 'mul':
            merged_output = output_fw_seq * output_bw_seq
        elif self.merge_mode == 'ave':
            merged_output = (output_fw_seq + output_bw_seq) / 2.0
        
        if self.return_sequences_bidir:
            return merged_output
        else:
            final_fw_state = output_fw_seq[:, -1, :] 
            final_bw_state = output_bw_seq[:, 0, :]  
            if self.merge_mode == 'concat':
                return np.concatenate((final_fw_state, final_bw_state), axis=-1)
            elif self.merge_mode == 'sum':
                return final_fw_state + final_bw_state
            elif self.merge_mode == 'mul':
                return final_fw_state * final_bw_state
            elif self.merge_mode == 'ave':
                return (final_fw_state + final_bw_state) / 2.0
            
class DenseLSTM(object):
    def __init__(self, units: int, input_size: int = None, activation_function: str = None):
        self.units = units
        self.input_size = input_size 
        self.activation_function = activation_function

        if self.activation_function:
            if self.activation_function.lower() == 'relu':
                self.act_forward = lambda x: np.maximum(0, x)
            elif self.activation_function.lower() == 'sigmoid':
                self.act_forward = lambda x: 1 / (1 + np.exp(-x))
            elif self.activation_function.lower() == 'softmax':
                self.act_forward = self._softmax
            elif self.activation_function.lower() == 'tanh':
                self.act_forward = np.tanh
            else:
                raise ValueError(f"Fungsi aktivasi tidak didukung: {self.activation_function}")
            
        self.W = np.zeros((self.input_size, self.units), dtype=np.float32)
        self.b = np.zeros((1, self.units), dtype=np.float32)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        linear_output = np.matmul(input_data, self.W) + self.b

        if self.activation_function:
            return self.act_forward(linear_output).astype(input_data.dtype)
        else:
            return linear_output.astype(input_data.dtype)
        
class ModelLSTM(object):
    def __init__(self, layers: list):
        if not isinstance(layers, list) or not all(hasattr(layer, 'forward') for layer in layers):
            raise ValueError("Parameter 'layers' harus berupa list dari objek lapisan yang memiliki metode 'forward'.")
        self.layers = layers

    def forward(self, input_data: np.ndarray) -> np.ndarray: 
        if tf.is_tensor(input_data): 
            input_data = input_data.numpy()
        
        current_output = input_data
        for i, layer in enumerate(self.layers):
            if isinstance(layer, DenseLSTM) and layer.W is None: 
                 layer._initialize_weights_if_needed(current_output.shape[-1])
            current_output = layer.forward(current_output)
        return current_output

    def load_weights_from_keras_model(self, keras_model_instance: tf.keras.Model):
        keras_weight_bearing_layers = [
            layer for layer in keras_model_instance.layers 
            if isinstance(layer, (tf.keras.layers.Bidirectional, 
                                   tf.keras.layers.SimpleRNN, 
                                   tf.keras.layers.Dense))
        ]
        custom_weight_bearing_layers = [
            layer for layer in self.layers
            if isinstance(layer, (BidirectionalLSTM, SimpleLSTM, DenseLSTM))
        ]

        if len(custom_weight_bearing_layers) != len(keras_weight_bearing_layers):
            print(
                f"PERINGATAN: Jumlah lapisan berbobot yang dapat dimuat tidak cocok! "
                f"Kustom: {len(custom_weight_bearing_layers)}, Keras: {len(keras_weight_bearing_layers)}."
            )
        
        for i, (custom_layer, keras_layer) in enumerate(zip(custom_weight_bearing_layers, keras_weight_bearing_layers)):
            try:
                if isinstance(custom_layer, BidirectionalLSTM) and isinstance(keras_layer, tf.keras.layers.Bidirectional):
                    if not (hasattr(keras_layer, 'forward_layer') and isinstance(keras_layer.forward_layer, tf.keras.layers.SimpleRNN) and hasattr(keras_layer, 'backward_layer') and isinstance(keras_layer.backward_layer, tf.keras.layers.SimpleRNN)):
                        print(f"PERINGATAN: Lapisan terbungkus dalam Keras Bidirectional ({keras_layer.name}) bukan SimpleRNN atau tidak memiliki forward/backward_layer yang diharapkan. Melewati.")
                        continue
                    
                    k_fw_rnn_weights = keras_layer.forward_layer.get_weights()
                    if len(k_fw_rnn_weights) == 3:
                        custom_layer.forward_rnn.U_t = k_fw_rnn_weights[0].T.astype(np.float32)
                        custom_layer.forward_rnn.W_t = k_fw_rnn_weights[1].astype(np.float32)
                        custom_layer.forward_rnn.b_xh = k_fw_rnn_weights[2].reshape(-1, 1).astype(np.float32)
                    else:
                        print(f"PERINGATAN: Jumlah bobot tidak terduga untuk forward_layer Keras Bidirectional ({len(k_fw_rnn_weights)}).")

                    k_bw_rnn_weights = keras_layer.backward_layer.get_weights()
                    if len(k_bw_rnn_weights) == 3:
                        custom_layer.backward_rnn.U_t = k_bw_rnn_weights[0].T.astype(np.float32)
                        custom_layer.backward_rnn.W_t = k_bw_rnn_weights[1].astype(np.float32)
                        custom_layer.backward_rnn.b_xh = k_bw_rnn_weights[2].reshape(-1, 1).astype(np.float32)
                    else:
                         print(f"PERINGATAN: Jumlah bobot tidak terduga untuk backward_layer Keras Bidirectional ({len(k_bw_rnn_weights)}).")
                
                elif isinstance(custom_layer, SimpleLSTM) and isinstance(keras_layer, tf.keras.layers.SimpleRNN):
                    k_weights = keras_layer.get_weights()
                    if len(k_weights) == 3: 
                        custom_layer.U_t = k_weights[0].T.astype(np.float32)
                        custom_layer.W_t = k_weights[1].astype(np.float32)
                        custom_layer.b_xh = k_weights[2].reshape(-1, 1).astype(np.float32)
                    else:
                        print(f"PERINGATAN: Jumlah bobot tidak terduga untuk Keras SimpleRNN ({len(k_weights)}).")
                
                elif isinstance(custom_layer, DenseLSTM) and isinstance(keras_layer, tf.keras.layers.Dense):
                    k_weights = keras_layer.get_weights()
                    if custom_layer.W is None:
                        inferred_input_size = k_weights[0].shape[0]
                        custom_layer._initialize_weights_if_needed(inferred_input_size)

                    if len(k_weights) == 2: 
                        custom_layer.W = k_weights[0].astype(np.float32)
                        custom_layer.b = k_weights[1].reshape(1, -1).astype(np.float32)
                    elif len(k_weights) == 1: 
                        custom_layer.W = k_weights[0].astype(np.float32)
                        custom_layer.b = np.zeros((1, custom_layer.units), dtype=np.float32)
                    else:
                        print(f"PERINGATAN: Jumlah bobot tidak terduga untuk Keras Dense ({len(k_weights)}).")
                else:
                    print(f"PERINGATAN PENTING: Tipe lapisan tidak cocok antara Kustom ({type(custom_layer).__name__}) dan Keras ({type(keras_layer).__name__}) yang difilter. Bobot tidak dimuat untuk pasangan ini.")
            except Exception as e:
                print(f"ERROR saat memuat bobot: {e}")
                import traceback
                traceback.print_exc()
        print("Selesai mencoba memuat bobot dari model Keras.")
