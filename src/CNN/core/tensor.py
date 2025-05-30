import numpy as np

from typing import List, Callable, Set, Union, Tuple, Optional

from ..functions import ActivationFunction, LossFunction


class Tensor:
    data: np.ndarray
    gradient: np.ndarray
    __children: List['Tensor']
    __op: str
    __backward: Callable[[], None]
    requires_grad: bool
    tensor_type: str

    def __init__(self, data: Union[np.ndarray, list, tuple, int, float],
                 _children: List['Tensor'] = [], _op: str = "", tensor_type: str = "", requires_grad: bool = True) -> None:
        
        if not isinstance(data, np.ndarray):
            try:
                data = np.array(data, dtype=float)
            except TypeError:
                raise TypeError(f"Expected np.ndarray or convertible type, but got {type(data).__name__} instead")
        
        self.data = data.astype(float)
        self.gradient = np.zeros_like(self.data, dtype=float)
        self.__children = list(_children)
        self.__op = _op
        self.__backward = lambda: None
        self.requires_grad = requires_grad
        self.tensor_type = tensor_type

    def __repr__(self) -> str:
        return (
            f"Tensor(data={self.data}, grad={self.gradient}, "
            f"op='{self.__op or 'None'}'"
            f"{', type=' + self.tensor_type if self.tensor_type else ''})"
        )

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    def add_x0(self) -> 'Tensor':
        """
        Adds x0 = 1 to each feature vector in the tensor.
        If input is 1D (features,), output is (1+features,).
        If input is 2D (batch, features), output is (batch, 1+features).
        """
        if self.data.ndim == 1:
            new_data = np.concatenate([np.array([1.0]), self.data])
        elif self.data.ndim == 2:
            batch_size = self.data.shape[0]
            ones_column = np.ones((batch_size, 1), dtype=float)
            new_data = np.concatenate([ones_column, self.data], axis=1)
        else:
            raise ValueError(f"add_x0 supports 1D or 2D data, but got {self.data.ndim}D data with shape {self.data.shape}")

        res = Tensor(new_data, [self], "add_x0")
        res.requires_grad = self.requires_grad

        def __backward():
            if self.data.ndim == 1:
                self.gradient += res.gradient[1:]
            elif self.data.ndim == 2:
                self.gradient += res.gradient[:, 1:]
        
        res.__backward = __backward
        return res

    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> 'Tensor':
        """
        Sums elements in the tensor into one element along the given axis.
        """
        res_data = np.sum(self.data, axis=axis, keepdims=keepdims)
        
        if not isinstance(res_data, np.ndarray):
            res_data = np.array([res_data]) 

        if res_data.ndim == 0:
            res_data = res_data.reshape(1,)

        res = Tensor(res_data, [self], f"sum(axis={axis}, keepdims={keepdims})")

        original_input_shape = self.data.shape

        def __backward():
            grad_to_propagate = res.gradient

            if not keepdims and axis is not None:
                axes_reduced_param = axis
                if isinstance(axes_reduced_param, int):
                    axes_reduced_param = (axes_reduced_param,)
                
                actual_reduced_axes = []
                if axes_reduced_param is not None:
                    for ax_val in axes_reduced_param:
                        actual_reduced_axes.append(ax_val % original_input_shape.ndim)
                    actual_reduced_axes.sort()

                temp_grad = grad_to_propagate
                for ax_to_insert in actual_reduced_axes:
                    temp_grad = np.expand_dims(temp_grad, axis=ax_to_insert)
                grad_to_propagate = temp_grad
            
            self.gradient += grad_to_propagate
        
        res.__backward = __backward
        return res

    @staticmethod
    def stack(tensors: List['Tensor'], axis: int = 0) -> 'Tensor':
        """
        Stacks a list of tensors along a new axis.
        All input tensors must have the same shape.
        """
        if not tensors:
            raise ValueError("Cannot stack an empty list of tensors.")
        
        first_shape = tensors[0].shape
        for t in tensors[1:]:
            if t.shape != first_shape:
                raise ValueError(f"All tensors to be stacked must have the same shape. Got {first_shape} and {t.shape}")

        stacked_data = np.stack([t.data for t in tensors], axis=axis)
        res = Tensor(stacked_data, tensors, f"stack(axis={axis})")

        def __backward():
            for i, child_tensor in enumerate(res.__children):
                slicer = [slice(None)] * res.gradient.ndim
                slicer[axis] = i
                child_tensor.gradient += res.gradient[tuple(slicer)]
        
        res.__backward = __backward
        return res

    def concat(self, tensors: List['Tensor'], axis: int = 0) -> 'Tensor':
        """
        Concatenates a list of tensors along an existing axis.
        All input tensors must have the same shape, except for the dimension
        corresponding to axis.
        """
        if not tensors:
            raise ValueError("Cannot concatenate an empty list of tensors.")

        tensors = [self] + tensors
        concatenated_data = np.concatenate([t.data for t in tensors], axis=axis)
        res = Tensor(concatenated_data, tensors, f"concatenate(axis={axis})")

        def __backward():
            current_offset = 0
            for child_tensor in res.__children:
                child_dim_size = child_tensor.shape[axis]
                
                slicer = [slice(None)] * res.gradient.ndim
                slicer[axis] = slice(current_offset, current_offset + child_dim_size)
                
                child_tensor.gradient += res.gradient[tuple(slicer)]
                current_offset += child_dim_size
        
        res.__backward = __backward
        return res

    def compute_loss(self, y_true: np.ndarray, loss_function: LossFunction) -> 'Tensor':
        """
        Computes the loss value using given loss function and y_true.
        """
        if not isinstance(y_true, np.ndarray):
            y_true = np.array(y_true, dtype=float)

        res_data = loss_function.forward(y_true, self.data)
        res = Tensor(np.array([res_data]).reshape(1,), [self], loss_function.__name__)

        original_self_data_shape = self.data.shape

        def __backward():
            grad_from_loss_fn = loss_function.backward(y_true, self.data)
            
            grad_for_self = grad_from_loss_fn
            if original_self_data_shape != grad_from_loss_fn.shape:
                grad_for_self = self.__unbroadcast_gradient(grad_from_loss_fn, original_self_data_shape)
            
            self.gradient += grad_for_self * res.gradient[0]
        
        res.__backward = __backward
        return res

    def compute_activation(self, activation_function: ActivationFunction) -> 'Tensor':
        """
        Computes the activation output using given activation function.
        """
        res_data = activation_function.forward(self.data)
        is_softmax_like = (hasattr(activation_function, '__name__') and 
                           "Softmax" in activation_function.__name__)
        res = Tensor(res_data, [self], activation_function.__name__)

        def __backward():
            grad_act_fn_output = activation_function.backward(self.data) 
            
            if is_softmax_like: 
                if self.data.ndim == 1 and grad_act_fn_output.ndim == 2 and res.gradient.ndim == 1:
                    self.gradient += grad_act_fn_output @ res.gradient
                elif self.data.ndim == 2 and grad_act_fn_output.ndim == 3 and res.gradient.ndim == 2:
                    self.gradient += np.einsum('bij,bj->bi', grad_act_fn_output, res.gradient)
                else:
                    raise NotImplementedError(
                        f"Softmax backward pass for data_ndim={self.data.ndim} (shape {self.data.shape}), "
                        f"jacobian_ndim={grad_act_fn_output.ndim} (shape {grad_act_fn_output.shape}), "
                        f"res_gradient_ndim={res.gradient.ndim} (shape {res.gradient.shape}) is not supported."
                    )
            else: 
                self.gradient += grad_act_fn_output * res.gradient
                
        res.__backward = __backward
        return res

    def backward(self) -> None:
        """
        Starts automated differentiation, calculating gradients from the root of operation tree all the way to the leaves.
        """
        topo: List[Tensor] = []
        visited: Set[Tensor] = set()

        def topological_sort(v_node: Tensor):
            if v_node not in visited:
                visited.add(v_node)
                for child in v_node.__children:
                    topological_sort(child)
                topo.append(v_node)
        
        topological_sort(self)

        self.gradient = np.ones_like(self.data, dtype=float)
        for v_node in reversed(topo):
            if v_node.requires_grad:
                v_node.__backward()

    def matmul(self, other: 'Tensor') -> 'Tensor':
        """
        Performs matrix multiplication between this tensor (self) and another tensor (other).
        Currently implemented for 2D x 2D matrix multiplication.
        self: (M, K), other: (K, N) => result: (M, N)
        """
        if not isinstance(other, Tensor):
            raise TypeError(f"Matrix multiplication requires a Tensor operand, got {type(other).__name__}")

        if self.ndim != 2 or other.ndim != 2:
            raise ValueError(f"Matrix multiplication currently supports 2D tensors only. "
                             f"Got shapes: {self.shape} and {other.shape}")

        if self.shape[1] != other.shape[0]:
            raise ValueError(f"Matrix multiplication dimension mismatch: "
                             f"{self.shape} @ {other.shape}. Inner dimensions must match.")

        
        res_data = self.data @ other.data 
        res = Tensor(res_data, [self, other], "matmul")

        def __backward():
            dL_dC = res.gradient

            if self.requires_grad:
                grad_A = dL_dC @ other.data.T
                self.gradient += grad_A

            if other.requires_grad:
                grad_B = self.data.T @ dL_dC
                other.gradient += grad_B
        
        res._Tensor__backward = __backward
        return res
    
    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        return self.matmul(other)

    def __rmatmul__(self, other: Union[np.ndarray, 'Tensor']) -> 'Tensor':
        if isinstance(other, np.ndarray):
            other_tensor = Tensor(other) 
            other_tensor.requires_grad = False 
            return other_tensor.matmul(self)
        elif isinstance(other, Tensor):
             return other.matmul(self)
        else:
            return NotImplemented

    def __unbroadcast_gradient(self, grad_term: np.ndarray, original_input_shape: tuple) -> np.ndarray:
        """
        Broadcasts gradient from a Tensor into its childrens when the dimensions between them are different.
        """
        current_grad = grad_term
        ndim_diff = current_grad.ndim - len(original_input_shape)
        if ndim_diff > 0:
            current_grad = np.sum(current_grad, axis=tuple(range(ndim_diff)))
        axes_to_sum = []
        for i, dim_orig in enumerate(original_input_shape):
            if i < current_grad.ndim and dim_orig == 1 and current_grad.shape[i] > 1:
                axes_to_sum.append(i)
        if axes_to_sum:
            current_grad = np.sum(current_grad, axis=tuple(axes_to_sum), keepdims=True)

        return current_grad.reshape(original_input_shape)

    def __add__(self, other: Union['Tensor', np.ndarray, float, int]) -> 'Tensor':
        if not isinstance(other, Tensor):
            other_tensor = Tensor(other, _op="const", tensor_type="const")
            other_tensor.requires_grad = False 
        else:
            other_tensor = other

        res_data = self.data + other_tensor.data
        res = Tensor(res_data, [self, other_tensor], "+")

        def __backward():
            grad_for_self = res.gradient
            if self.data.shape != res.data.shape:
                grad_for_self = self.__unbroadcast_gradient(res.gradient, self.data.shape)
            self.gradient += grad_for_self
            if other_tensor.requires_grad:
                grad_for_other = res.gradient
                if other_tensor.data.shape != res.data.shape:
                    grad_for_other = self.__unbroadcast_gradient(res.gradient, other_tensor.data.shape)
                other_tensor.gradient += grad_for_other

        res.__backward = __backward
        return res
    
    def __mul__(self, other: Union['Tensor', np.ndarray, float, int]) -> 'Tensor':
        if not isinstance(other, Tensor):
            other_tensor = Tensor(other, _op="const", tensor_type="const")
            other_tensor.requires_grad = False
        else:
            other_tensor = other

        res_data = self.data * other_tensor.data
        res = Tensor(res_data, [self, other_tensor], "*")

        def __backward():
            grad_self_term = other_tensor.data * res.gradient
            if self.data.shape != res.data.shape:
                grad_self_term = self.__unbroadcast_gradient(grad_self_term, self.data.shape)

            self.gradient += grad_self_term
            if other_tensor.requires_grad:
                grad_other_term = self.data * res.gradient
                if other_tensor.data.shape != res.data.shape:
                    grad_other_term = self.__unbroadcast_gradient(grad_other_term, other_tensor.data.shape)

                other_tensor.gradient += grad_other_term

        res.__backward = __backward
        return res

    def __pow__(self, exponent: Union[int, float]) -> 'Tensor':
        if not isinstance(exponent, (int, float)):
            raise TypeError(f"Exponent must be int or float, but got {type(exponent).__name__}")
        
        res_data = self.data ** exponent
        res = Tensor(res_data, [self], f'**{exponent}')

        def __backward():
            self.gradient += (exponent * (self.data**(exponent - 1.0))) * res.gradient

        res.__backward = __backward
        return res

    def __neg__(self) -> 'Tensor':
        return self * -1.0

    def __radd__(self, other: Union['Tensor', np.ndarray, float, int]) -> 'Tensor':
        return self + other

    def __iadd__(self, other: Union['Tensor', np.ndarray, float, int]) -> 'Tensor':
        return self + other

    def __sub__(self, other: Union['Tensor', np.ndarray, float, int]) -> 'Tensor':
        return self + (-other)

    def __rsub__(self, other: Union['Tensor', np.ndarray, float, int]) -> 'Tensor':
        if not isinstance(other, Tensor):
            other_tensor = Tensor(other, _op="const", tensor_type="const")
            other_tensor.requires_grad = False
            return other_tensor + (-self)
        
        return other + (-self)
    
    def __isub__(self, other: Union['Tensor', np.ndarray, float, int]) -> 'Tensor':
        return self - other 

    def __rmul__(self, other: Union['Tensor', np.ndarray, float, int]) -> 'Tensor':
        return self * other
    
    def __imul__(self, other: Union['Tensor', np.ndarray, float, int]) -> 'Tensor':
        return self * other

    def __truediv__(self, other: Union['Tensor', np.ndarray, float, int]) -> 'Tensor':
        if not isinstance(other, Tensor):
            other_tensor = Tensor(other, _op="const", tensor_type="const")
            other_tensor.requires_grad = False
            return self * (other_tensor ** -1.0)
        
        return self * (other ** -1.0)

    def __rtruediv__(self, other: Union['Tensor', np.ndarray, float, int]) -> 'Tensor':
        if not isinstance(other, Tensor):
            other_tensor = Tensor(other, _op="const", tensor_type="const")
            other_tensor.requires_grad = False
            return other_tensor * (self ** -1.0)
        
        return other * (self ** -1.0)
    
    def __array__(self, dtype=None) -> np.ndarray:
        return self.data if dtype is None else self.data.astype(dtype)

    def reshape(self, *new_shape: int) -> 'Tensor':
        if self.data.size != np.prod(new_shape):
            raise ValueError(f"Cannot reshape array of size {self.data.size} into shape {new_shape}. Target product: {np.prod(new_shape)}")
        
        res_data = self.data.reshape(*new_shape)
        res = Tensor(res_data, [self], "reshape")
        original_shape = self.data.shape

        def __backward():
            self.gradient += res.gradient.reshape(original_shape)

        res.__backward = __backward
        return res

    def transpose(self, axes: Optional[Tuple[int, ...]] = None) -> 'Tensor':
        res_data = np.transpose(self.data, axes)
        res = Tensor(res_data, [self], f"transpose(axes={axes})")

        def __backward():
            if axes is None:
                inv_axes = None
            else:
                inv_axes = tuple(np.argsort(axes))
            self.gradient += np.transpose(res.gradient, inv_axes)

        res.__backward = __backward
        return res

    def mean(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> 'Tensor':
        res_data = np.mean(self.data, axis=axis, keepdims=keepdims)
        res = Tensor(res_data, [self], f"mean(axis={axis}, keepdims={keepdims})")
        original_shape = self.data.shape
        output_shape = res_data.shape

        def __backward():
            if axis is None:
                N = self.data.size
            else:
                if isinstance(axis, int):
                    N = original_shape[axis]
                else:
                    N = np.prod([original_shape[i] for i in axis])

            grad_to_distribute = res.gradient
            if not keepdims and axis is not None:
                current_shape = list(output_shape)
                axes_to_insert = sorted(list(axis) if isinstance(axis, tuple) else [axis])
                for ax_idx in axes_to_insert:
                    current_shape.insert(ax_idx, 1)
                grad_to_distribute = grad_to_distribute.reshape(tuple(current_shape))

            self.gradient += (1/N) * grad_to_distribute

        res.__backward = __backward
        return res