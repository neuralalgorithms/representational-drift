# -*- coding: utf-8 -*-
"""
HebbianModels.py
Base implementations of Hebbian Learning Models including Oja and Sanger networks.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Any
import os

class HebbianModel(ABC):
    """
    Base class for linear Hebbian models.

    Attributes:
        input_size (int): Number of input dimensions
        output_size (int): Number of output neurons
        learning_rate (float): Learning rate for weight updates
        weights (np.ndarray): Weight matrix of shape (output_size, input_size)

    Conventions:
        - x: shape (samples, n); Input samples
        - weights: shape (k, n); Weight matrix where k is output_size
        - y = weights @ x.T: shape (k,); Output after forward pass
    """
    def __init__(self, input_size: int, output_size: int, learning_rate: float = 0.01, rng: Optional[Any] = None) -> None:
        if input_size <= 0 or output_size <= 0:
            raise ValueError(f"input_size ({input_size}) and output_size ({output_size}) must be positive integers.")
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.learning_rate = float(learning_rate)
        self.rng = np.random.default_rng(rng)
        # Initialize weights with small random values
        self.weights = 0.01 * self.rng.standard_normal((self.output_size, self.input_size))


    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the forward pass of the network.

        Args:
            x: Input sample of shape (n,) or (1, n)

        Returns:
            np.ndarray: Output vector of shape (k,)

        Raises:
            ValueError: If input dimensions don't match the model's input size
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.shape[0] != self.input_size:
            raise ValueError(f"Input shape {x.shape} does not match model input size ({self.input_size},)")
        return self.weights @ x.T  # (k,)

    @abstractmethod
    def step(self, x: np.ndarray) -> np.ndarray:
        """
        Perform one online update with a single sample.

        Args:
            x: Input sample of shape (n,)

        Returns:
            np.ndarray: Output vector of shape (k,) after weight update

        Note:
            This is an abstract method that must be implemented by subclasses.
            Implementation must:
            1. Compute y = forward(x)
            2. Update self.weights in-place according to learning rule
            3. Return the output y
        """
        raise NotImplementedError

    def train(self, X, epochs=1, shuffle=True):
        """
        Online training over X for 'epochs' passes.
        X: array-like of shape (sample, n)
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.input_size:
            raise ValueError(f"X has shape {X.shape}, expected (N, {self.input_size})")
        N = X.shape[0]
        for _ in range(int(epochs)):
            idx = np.arange(N)
            if shuffle:
                self.rng.shuffle(idx)
            for t, i in enumerate(idx):
                y = self.step(X[i]) 

    def normalize_weights(self, axis: int = 1, epsilon: float = 1e-12) -> None:
            """
            Normalize weights along specified axis.

            Args:
                axis (int): Axis along which to normalize (0=columns, 1=rows)
                epsilon (float): Small constant for numerical stability
            """
            norms = np.linalg.norm(self.weights, axis=axis, keepdims=True) + epsilon
            self.weights /= norms

    def get_summary(self) -> dict:
        """
        Get a summary of the model's architecture and parameters.

        Returns:
            dict: Model summary including input/output sizes and parameter count
        """
        return {
            "model_type": self.__class__.__name__,
            "input_size": self.input_size,
            "output_size": self.output_size,
            "learning_rate": self.learning_rate,
            "parameter_count": self.weights.size,
            "weight_shape": self.weights.shape
        }

class OjaNetwork(HebbianModel):
    """
    Implementation of Oja's learning rule for principal component analysis.
    
    The network learns the principal component subspace through online updates:
    W <- W + η(yx^T - yy^TW)
    
    Where:
    - y = Wx is the output
    - η is the learning rate
    - W is the weight matrix
    
    This implementation can learn multiple principal components simultaneously,
    though their order is not enforced.
    """

    def step(self, x):
        x = np.asarray(x, dtype=float).reshape(-1)           # (n,)
        y = self.forward(x)                                  # (k,)
        # ΔW = eta * ( y x^T - y y^T w )                   # yx^T is (k, n); y y^T W is (k, n)
        self.weights += self.learning_rate * (np.outer(y, x) - np.outer(y, y) @ self.weights)
        return y
    

class SangerNetwork(HebbianModel):
    """
    Implementation of Sanger's rule (Generalized Hebbian Algorithm).
    
    The network learns ordered principal components through online updates using
    the learning rule:
        ΔW = η(yx^T - LT(yy^T)W)
    
    Where:
    - LT() is the lower triangular operator that zeros out upper triangle
    - y = Wx is the output
    - η is the learning rate
    - W is the weight matrix
    
    Unlike Oja's rule, this implementation enforces ordering of principal components,
    with each output neuron learning a different PC in descending order of variance.
    The first neuron converges to PC1, second to PC2, and so on.
    
    References:
        Sanger, T.D. (1989) Optimal unsupervised learning in a single-layer 
        linear feedforward neural network. Neural Networks, 2(6), 459-473.
    """

    def step(self, x):
        x = np.asarray(x, dtype=float).reshape(-1)
        y = self.forward(x)
        # ΔW = eta * (yx^T - BW), B is the lower triangular matrix of y y^T
        B = np.tril(np.outer(y, y))
        self.weights += self.learning_rate * (np.outer(y,x) - B @ self.weights)
        return y