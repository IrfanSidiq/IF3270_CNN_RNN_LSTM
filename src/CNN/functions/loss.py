import numpy as np

from abc import ABC, abstractmethod


class LossFunction(ABC):
    @abstractmethod
    def forward(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Computes loss value using loss function.
        """
        pass

    @abstractmethod
    def backward(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Computes gradient of loss function.
        """
        pass


class MeanSquaredError(LossFunction):
    @staticmethod
    def forward(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def backward(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        diff = y_pred - y_true 
        num_elements_in_mean = diff.size 
        if num_elements_in_mean == 0: return np.zeros_like(diff)
        return (2 / num_elements_in_mean) * diff

class BinaryCrossEntropy(LossFunction):
    @staticmethod
    def forward(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        epsilon = 1e-9
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        term_before_mean = y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped)
        return -np.mean(term_before_mean)

    @staticmethod
    def backward(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        epsilon = 1e-9
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        
        grad_numerator = y_pred_clipped - y_true 
        grad_denominator = y_pred_clipped * (1 - y_pred_clipped)
        
        grad_denominator = np.where(grad_denominator == 0, epsilon, grad_denominator)

        raw_grad = grad_numerator / grad_denominator
        num_elements_in_mean = raw_grad.size
        if num_elements_in_mean == 0: return np.zeros_like(raw_grad)
        
        return (1 / num_elements_in_mean) * raw_grad

class CategoricalCrossEntropy(LossFunction):
    @staticmethod
    def forward(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        epsilon = 1e-9
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        term_before_mean = y_true * np.log(y_pred_clipped)
        return -np.mean(term_before_mean)

    @staticmethod
    def backward(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        epsilon = 1e-9
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        y_pred_clipped_safe = np.where(y_pred_clipped == 0, epsilon, y_pred_clipped)

        raw_grad = -y_true / y_pred_clipped_safe
        num_elements_in_mean = raw_grad.size
        if num_elements_in_mean == 0: return np.zeros_like(raw_grad)
        
        return (1 / num_elements_in_mean) * raw_grad