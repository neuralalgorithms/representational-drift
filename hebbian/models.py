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
    def __init__(self, output_size: int, 
                 learning_rate: float = 0.01, 
                 learning_rate2: Optional[float] = None,
                 rng: Optional[Any] = None) -> None:
        self.output_size = int(output_size)
        self.learning_rate = float(learning_rate)
        if learning_rate2 is None:
            self.learning_rate2 = float(learning_rate)
        else:
            self.learning_rate2 = float(learning_rate2)
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
        self.W += self.learning_rate * np.outer(y, x) - self.learning_rate2* np.outer(y, y) @ self.W
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
        self.W += self.learning_rate * np.outer(y,x) - self.learning_rate2 * B @ self.W
        return y

class LeenCompletePCA:
    """
    Implementation of Leen's "complete" PCA model (Section 4).

    In this model, the neuron's output `y` is calculated recurrently,
    meaning the lateral connections `q` (eta in the paper) directly
    influence the cell's response in a feedback loop.
 
    Equations of Motion (ensemble average form):
    -------------------------------------------
    Forward Pass (Eq. 33):
        y_{t+1} = W(t) x_t + q(t) y_t
        (iterated for a fixed number of settling steps)

    Forward Weight Update (Eq. 35):
        Δw = η_w * ( <y x^T> - Diag(<y y^T>) w )

    Lateral Connection Update (Activity-Dependent, Eq. 47):
        Δq_ik = η_v * ( (<y_i^2> + <y_k^2>) * v_ik - C * <y_i y_k> )

    Parameters
    ----------
    input_dim : int
    output_dim : int
    eta_w : float             # Forward learning rate
    eta_v : float             # Lateral learning rate (make this larger than eta_w)
    C : float                 # Coupling; pick > 1 (e.g., 1.5)
    ema_alpha : float         # EMA step for activity λ_i ≈ E[y_i^2]
    symmetrize_V : bool       # Keep V symmetric + zero diagonal after each step
    seed : int or None
    """
    def __init__(self,
                 input_dim: int, output_dim: int,
                 eta_w: float = 1e-3,
                 eta_v: float = 1e-2,
                 C: float = 1.5,
                 ema_alpha: float = 0.05,
                 symmetrize_V: bool = True,
                 seed: int | None = 0, 
                 settling_steps: int = 10, 
                 noise_level: float = 0.01, 
                 clip_V_spectral: float = 0.95):   # <-- NEW
        self.d = input_dim
        self.m = output_dim
        self.eta_w = eta_w
        self.eta_v = eta_v
        self.C = C
        self.ema_alpha = ema_alpha
        self.symmetrize_V = symmetrize_V
        self.settling_steps = settling_steps
        self.noise_level = noise_level
        self.clip_V_spectral = clip_V_spectral

        self.rng = np.random.default_rng(seed)
        W = self.rng.normal(size=(self.m, self.d))
        W /= np.linalg.norm(W, axis=1, keepdims=True) + 1e-12 
        self.W = W
        self.initial_W = W.copy()
        # Symmetric zero-diagonal init for V
        self.V = np.zeros((self.m, self.m))
        # EMA of activities (start small positive to avoid zero)
        self.lam = np.full(self.m, 1e-6)
        # initialize the y_0. with shape (batch_size, m)

    # ---------- helpers ----------
    # ---------- NEW: reset ----------
    def reset(self):
        W = self.rng.normal(size=(self.m, self.d))
        W /= np.linalg.norm(W, axis=1, keepdims=True) + 1e-12
        self.W = W
        self.initial_W = W.copy()

        # Reinitialize lateral q and EMA lam exactly as in __init__
        self.V = np.zeros((self.m, self.m))
        self.lam = np.full(self.m, 1e-6)

    # ---------- modified forward function to Y = Wx + Qy ----------
    def _forward(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates the cell response by iteratively simulating the fast dynamics.
        y_{t+1} = W(t) x_t + q(t) * y_t
        """
        # Line 1: Calculate the constant feed-forward input drive, Z = Wx
        # This is the external input to the recurrent system.
        Z = X @ self.W.T
        # Line 2: Initialize the neuron activities (y_0) to zero at the start.
        # Before the input arrives, the neurons are assumed to be silent. 
        Y = self.rng.normal(scale=self.noise_level, size=Z.shape)
        # Y = np.zeros_like(Z)
        # Line 3: Loop for a fixed number of steps to simulate settling.
        # This explicitly models the passage of time on the fast (millisecond) scale.
        
        for _ in range(self.settling_steps):
            # Line 4: Calculate the next state of neural activity.
            # Y_t+1 = Z (feed-forward) + Y_t @ q.T (recurrent feedback)
            # The current activity Y is fed back through the lateral connections q
            # and added to the constant external drive Z.
            Y = Z + Y @ self.V.T
        # Line 5: Return the final, settled activity after the loop.
        return Y

    def _symmetrize_V(self):
        self.V = 0.5 * (self.V + self.V.T)
        np.fill_diagonal(self.V, 0.0)

    def _clip_V_spectral_norm(self):
        if self.clip_V_spectral is None:
            return
        # Spectral norm via SVD
        u, s, vt = np.linalg.svd(self.V, full_matrices=False)
        smax = s[0]
        if smax > self.clip_V_spectral:
            s = s * (self.clip_V_spectral / (smax + 1e-12))
            self.V = (u * s) @ vt
            self._symmetrize_V()
    # ---------- public API ----------

    def step(self, X: np.ndarray):
        """
        One learning step on a batch X: shape (B, d).
        """
        X = np.asarray(X, dtype=float)
        B = X.shape[0]

        # Forward pass (recurrent calculation)
        Y = self._forward(X)  # (B, m)
        # Batch moments from the complete response
        y_cov = (Y.T @ Y) / B      # (m, m) ~ <y y^T>
        y_var = np.diag(y_cov).copy()  # (m,)   ~ <y_i^2>

        # EMA update of activities λ_i ~ E[y_i^2]
        self.lam = (1.0 - self.ema_alpha) * self.lam + self.ema_alpha * y_var

        # ----- Lateral update (Activity-Dependent Anti-Hebbian) -----
        lam_sum = self.lam[:, None] + self.lam[None, :]  # (m, m) (λ_i + λ_j)
        # Implements Δq_ij = η_q * ( (λ_i + λ_j) * q_ij - C * <y_i y_j> )
        dV = self.eta_v * (lam_sum * self.V - self.C * y_cov)
        np.fill_diagonal(dV, 0.0)
        self.V += dV

        if self.symmetrize_V:
            self._symmetrize_V()
        self._clip_V_spectral_norm()

        # ----- Forward update (Hebb-Oja on complete response) -----
        # Implements Δw = η_w * ( <y x^T> - Diag(<y y^T>) w )
        hebb_term = (Y.T @ X) / B  # (m, d)
        oja_term = y_var[:, None] * self.W      # (m, d)
        dW = self.eta_w * (hebb_term - oja_term)
        self.W += dW

    def transform(self, X: np.ndarray) -> np.ndarray:
        "Project X onto learned components using the recurrent dynamics."
        return self._forward(np.asarray(X, dtype=float))

    @property
    def components_(self) -> np.ndarray:
        "Rows of W are the learned components. Shape: (m, d)"
        # Normalize for stability and consistency, as Oja's rule keeps them near norm 1
        W_norm = self.W.copy()
        W_norm /= np.linalg.norm(W_norm, axis=1, keepdims=True) + 1e-12

        return W_norm

    @property
    def initial_weights_(self) -> np.ndarray:
        "Initial weights W, make sure the model is re-initialized. Shape: (m, d)"
        return self.initial_W

class LeenMinimalPCA:
    """
    Activity-dependent Leen/Földiak PCA learner (NumPy).

    y = (I - q)^{-1} W x        # complete mode
    or y = W x                  # minimal mode (fast, robust for learning)

    Lateral (off-diagonal only):
        q_{ij} <- q_{ij} + η_q * (λ_i + λ_j) * ( - q_{ij} - C * <y_i y_j> )

    Forward (per unit i):
        w_i <- w_i + η_W * ( < x * s_i > - < y_i^2 > w_i )
        where s = y + q y  (vector per sample); i.e. s = (I + q) y

    Parameters
    ----------
    input_dim : int
    output_dim : int
    eta_w : float            # forward learning rate
    eta_q : float            # lateral learning rate (make this larger than eta_w)
    C : float                # coupling; pick > 1 (e.g., 1.5)
    ema_alpha : float        # EMA step for activity λ_i ≈ E[y_i^2]
    forward_mode : str       # "minimal", "complete", or "complete_first_order"
    symmetrize_q : bool      # keep q explicitly symmetric + zero diagonal after each step
    clip_q_spectral : float or None  # if not None, shrink q to keep ||q||_2 <= this value (e.g., 0.95)
    seed : int or None
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 eta_w: float = 1e-3,
                 eta_q: float = 1e-2,
                 C: float = 1.5,
                 ema_alpha: float = 0.05,
                 symmetrize_q: bool = True,
                 seed: int | None = 0, 
                 clip_q_spectral: float = 0.95):

        self.d = input_dim
        self.m = output_dim
        self.eta_w = eta_w
        self.eta_q = eta_q
        self.C = C
        self.ema_alpha = ema_alpha
        self.symmetrize_q = symmetrize_q
        self.clip_q_spectral = clip_q_spectral

        rng = np.random.default_rng(seed)
        # Row-wise unit-norm init for W
        W = rng.normal(size=(self.m, self.d))
        W /= np.linalg.norm(W, axis=1, keepdims=True) + 1e-12
        self.W = W
        # Symmetric zero-diagonal init for q
        self.q = np.zeros((self.m, self.m))
        np.fill_diagonal(self.q, 0.0)
        # EMA of activities (start small positive to avoid zero)
        self.lam = np.full(self.m, 1e-6)

    # ---------- helpers ----------

    def _forward(self, X: np.ndarray) -> np.ndarray:
        """
        X: (B, d) -> Y: (B, m)
        """
        Y = X @ self.W.T  # (B, m) = W x
        return Y

    def _symmetrize_q(self):
        self.q = 0.5 * (self.q + self.q.T)
        np.fill_diagonal(self.q, 0.0)

    def _clip_q_spectral_norm(self):
        if self.clip_q_spectral is None:
            return
        # Spectral norm via SVD
        u, s, vt = np.linalg.svd(self.q, full_matrices=False)
        smax = s[0]
        if smax > self.clip_q_spectral:
            s = s * (self.clip_q_spectral / (smax + 1e-12))
            self.q = (u * s) @ vt
            self._symmetrize_q()    


    # ---------- public API ----------

    def step(self, X: np.ndarray):
        """
        One learning step on a batch X: shape (B, d).
        """
        X = np.asarray(X, dtype=float)
        B = X.shape[0]

        # Forward pass
        Y = self._forward(X)                                     # (B, m)

        # Batch moments
        y_cov = (Y.T @ Y) / B                                    # (m, m)  ~ < y_i y_j >
        y_var = np.diag(y_cov).copy()                            # (m,)

        # EMA update of activities λ_i ~ E[y_i^2]
        self.lam = (1.0 - self.ema_alpha) * self.lam + self.ema_alpha * y_var

        # ----- Lateral update (off-diagonal) -----
        lam_sum = self.lam[:, None] + self.lam[None, :]          # (m, m)  (λ_i + λ_j)
        # dq = self.eta_q * lam_sum * ( - self.q - self.C * y_cov )  # 09222025: This is wrong. Fixed below:
        dq = self.eta_q * (- lam_sum * self.q  - self.C * y_cov)             # decay term
        np.fill_diagonal(dq, 0.0)
        self.q += dq

        if self.symmetrize_q:
            self._symmetrize_q()
        self._clip_q_spectral_norm()
        # ----- Forward update (Hebb-Oja, gated by q) -----
        # s = (I + q) y  (per sample); with row-vectors: S = Y @ (I+q)^T
        S = Y @ (np.eye(self.m) + self.q).T                      # (B, m)
        # < x * s_i > as a matrix: (d, m) then transpose to (m, d)
        XS = (X.T @ S) / B                                       # (d, m)
        dW = self.eta_w * (XS.T - y_var[:, None] * self.W)       # (m, d)
        self.W += dW

        # Mild row renorm to avoid drift (Oja already stabilizes norms)
        row_norms = np.linalg.norm(self.W, axis=1, keepdims=True) + 1e-12
        self.W /= row_norms

    def transform(self, X: np.ndarray) -> np.ndarray:
        "Project X onto learned components (uses the current forward mode)."
        return self._forward(np.asarray(X, dtype=float))

    @property
    def components_(self) -> np.ndarray:
        "Rows of W are the learned components (unit-norm). Shape: (m, d)"
        return self.W.copy()

