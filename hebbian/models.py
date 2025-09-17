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
        - W: shape (m, n); Weight matrix where m is output_size
        - y = weights @ x.T: shape (m,); Output after forward pass
    """
    def __init__(self, output_size: int, learning_rate: float = 0.01, rng: Optional[Any] = None) -> None:
        self.output_size = int(output_size)
        self.learning_rate = float(learning_rate)
        self.rng = np.random.default_rng(rng)
        # Initialize weights with small random values
        self.input_size = None  # to be set on first forward call


    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the forward pass of the network.

        Args:
            x: Input sample of shape (n,) or (1, n)
        Returns:
            np.ndarray: Output vector of shape (m,)
        Raises:
            ValueError: If input dimensions don't match the model's input size
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        if self.input_size is None:
            self.input_size = x.shape[0]
            # Initialize weights if not already done
            self.W = 0.01 * self.rng.standard_normal((self.output_size, self.input_size))
        elif x.shape[0] != self.input_size:
            raise ValueError(f"Input dimension {x.shape[0]} does not match model input size {self.input_size}")
        return self.W @ x.T  # (m,)

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

    def train(self, X):
        """
        A very simple trial by trial online training over X.
        X: array-like of shape (sample, n)
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"Input X must be 2D, got shape {X.shape}")
        N = X.shape[0]
        for i in range(N):
            y = self.step(X[i]) 

    def normalize_weights(self, axis: int = 1, epsilon: float = 1e-12) -> None:
            """
            Normalize weights along specified axis.

            Args:
                axis (int): Axis along which to normalize (0=columns, 1=rows)
                epsilon (float): Small constant for numerical stability
            """
            norms = np.linalg.norm(self.W, axis=axis, keepdims=True) + epsilon
            self.W /= norms

    def get_effective_weights(self):
        return self.W

    def get_summary(self) -> dict:
        """
        Get a summary of the model's architecture and parameters.

        Returns:
            dict: Model summary including input/output sizes and parameter count
        """
        return {
            "model_type": self.__class__.__name__,
            "output_size": self.output_size,
            "learning_rate": self.learning_rate,
            "parameter_count": self.W.size,
            "weight_shape": self.W.shape
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
        self.W += self.learning_rate * (np.outer(y, x) - np.outer(y, y) @ self.W)
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
        self.W += self.learning_rate * (np.outer(y,x) - B @ self.W)
        return y
