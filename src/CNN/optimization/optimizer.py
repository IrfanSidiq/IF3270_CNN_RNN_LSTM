import numpy as np

from abc import ABC, abstractmethod
from typing import List, Dict

from ..core import Tensor


class Optimizer(ABC):
    parameters: List[Tensor]
    learning_rate: float

    def __init__(self, learning_rate: float = 0.01) -> None:
        self.learning_rate = learning_rate
        self.parameters = None

    def set_parameters(self, parameters: List[Tensor]):
        """
        Loads model's parameter into this optimizer.
        """
        self.parameters = parameters

    @abstractmethod
    def step(self) -> None:
        """
        Updates parameters based on learning rate.
        """
        pass

    def zero_grad(self) -> None:
        """
        Sets all parameters' gradient to zero.
        """
        if self.parameters is None:
            raise RuntimeError(f"Parameters has not been set yet! Set the parameters to optimize using set_parameters().")

        for param in self.parameters:
            param.gradient.fill(0)


class Adam(Optimizer):
    def __init__(self,
                 learning_rate: float = 0.001,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-8):
        
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        self.m: Dict[int, np.ndarray] = {}
        self.v: Dict[int, np.ndarray] = {}
        self.t: int = 0

    def set_parameters(self, parameters: List[Tensor]):
        super().set_parameters(parameters)

        if self.parameters:
            for param in self.parameters:
                if param.requires_grad:
                    param_id = id(param)
                    self.m[param_id] = np.zeros_like(param.data, dtype=float)
                    self.v[param_id] = np.zeros_like(param.data, dtype=float)
        
        self.t = 0

    def step(self) -> None:
        if self.parameters is None:
            raise RuntimeError("Parameters have not been set yet! Set the parameters to optimize using set_parameters().")

        self.t += 1

        for param in self.parameters:
            if not param.requires_grad:
                continue

            param_id = id(param)
            grad = param.gradient

            self.m[param_id] = self.beta1 * self.m[param_id] + (1 - self.beta1) * grad
            self.v[param_id] = self.beta2 * self.v[param_id] + (1 - self.beta2) * (grad ** 2)
            
            m_hat = self.m[param_id] / (1 - self.beta1 ** self.t)
            v_hat = self.v[param_id] / (1 - self.beta2 ** self.t)

            param.data -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)