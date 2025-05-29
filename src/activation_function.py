from abc import ABC, abstractmethod
import numpy as np
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
    def forward(x: np.ndarray) -> np.ndarray:
        x_shifted = x - np.max(x)
        sum_e = np.exp(x_shifted).sum()
        return np.exp(x_shifted) / sum_e
    
    @staticmethod
    def backward(x: np.ndarray) -> np.ndarray:
        softmax_x = Softmax.forward(x)
        size = x.size
        arr = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                delta = 1 if i == j else 0
                arr[i, j] = softmax_x[i] * (delta - softmax_x[j])
        
        return arr
                

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