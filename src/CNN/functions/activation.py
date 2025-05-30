import numpy as np

from abc import ABC, abstractmethod
from scipy.special import erf


class ActivationFunction(ABC):
    @abstractmethod
    def forward(x: np.ndarray) -> np.ndarray:
        """
        Computes activation output using activation function.
        """
        pass

    @abstractmethod
    def backward(x: np.ndarray) -> np.ndarray:
        """
        Computes gradient of activation function.
        """
        pass


class Linear(ActivationFunction):
    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        return x
    
    @staticmethod
    def backward(x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)


class ReLU(ActivationFunction):
    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    @staticmethod
    def backward(x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)


class Sigmoid(ActivationFunction):
    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def backward(x: np.ndarray) -> np.ndarray:
        sigmoid_x = Sigmoid.forward(x)
        return sigmoid_x * (1 - sigmoid_x)
    
class HyperbolicTangent(ActivationFunction):
    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
    
    @staticmethod
    def backward(x: np.ndarray) -> np.ndarray:
        return np.power((2 / (np.exp(x) - np.exp(-x))), 2)

class Softmax(ActivationFunction):
    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray: 
        if x.ndim == 1:
            x_shifted = x - np.max(x) 
            exps = np.exp(x_shifted)
            return exps / np.sum(exps)
        elif x.ndim == 2:
            x_shifted = x - np.max(x, axis=1, keepdims=True)
            exps = np.exp(x_shifted)
            return exps / np.sum(exps, axis=1, keepdims=True)
        else:
            raise ValueError("Softmax input must be 1D or 2D.")

    @staticmethod
    def backward(x: np.ndarray) -> np.ndarray: 
        if x.ndim == 1:
            s = Softmax.forward(x) 
            num_classes = s.shape[0]
            
            jacobian = np.zeros((num_classes, num_classes), dtype=float)
            for i in range(num_classes):
                for j in range(num_classes):
                    delta = 1 if i == j else 0
                    jacobian[i, j] = s[i] * (delta - s[j])
            return jacobian 

        elif x.ndim == 2:
            batch_size, num_classes = x.shape
            s_batch = Softmax.forward(x) 

            batched_jacobian = np.zeros((batch_size, num_classes, num_classes), dtype=float)
            
            for b in range(batch_size):
                s_sample = s_batch[b, :] 
                jacobian_sample = np.zeros((num_classes, num_classes), dtype=float)
                for i in range(num_classes):
                    for j in range(num_classes):
                        delta = 1 if i == j else 0
                        jacobian_sample[i, j] = s_sample[i] * (delta - s_sample[j])
                batched_jacobian[b, :, :] = jacobian_sample
            return batched_jacobian 
        else:
            raise ValueError("Softmax.backward input x must be 1D or 2D.")
                
class GELU(ActivationFunction):
    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        return x/2 * (1 + erf(x + np.sqrt(2)))
    
    @staticmethod
    def backward(x: np.ndarray) -> np.ndarray:
        d_phi = np.exp(-np.power(x,2) / 2) / np.sqrt(2 * np.pi)
        phi = 0.5 * (1 + erf(x + np.sqrt(2)))
        return x * d_phi + phi
    
class SILU(ActivationFunction):
    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        return x * Sigmoid.forward(x)
    
    @staticmethod
    def backward(x: np.ndarray) -> np.ndarray:
        return Sigmoid.forward(x) + x * Sigmoid.backward(x) * (1 - Sigmoid.forward(x))