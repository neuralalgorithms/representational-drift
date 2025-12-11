# -*- coding: utf-8 -*-
"""
HebbianModels.py
Base implementations of Hebbian Learning Models including Oja and Sanger networks.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Any
import os

### ---------   Hebbian models without lateral connections --------- ###
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
    def __init__(self,
                 output_size: int, 
                 learning_rate: float = 0.01, 
                 learning_rate2: Optional[float] = None,
                 input_size: int = None,
                 rng: Optional[Any] = None) -> None:
        self.output_size = int(output_size)
        self.learning_rate = float(learning_rate)
        self.rng = np.random.default_rng(rng)
        # Initialize weights with small random values
        if input_size is None:
            self.input_size = None
        else:
            self.input_size = int(input_size)
            self.W = 0.01 * self.rng.standard_normal((self.output_size, self.input_size))

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

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project X onto learned components.
        
        Args:
            X: Input samples of shape (n_samples, n_features) or (n_features,)
        Returns:
            np.ndarray: Output of shape (n_samples, output_size) or (output_size,)
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            # Single sample
            return self.forward(X)
        elif X.ndim == 2:
            # Batch of samples: (n_samples, n_features)
            if self.input_size is None:
                self.input_size = X.shape[1]
                self.W = 0.01 * self.rng.standard_normal((self.output_size, self.input_size))
            # Project each sample: W @ X.T gives (output_size, n_samples)
            # Transpose to get (n_samples, output_size)
            return (self.W @ X.T).T
        else:
            raise ValueError(f"X must be 1D or 2D, got shape {X.shape}")
    
    def reset(self):
        """
        Reset the model to initial random weights.
        Useful for multiple training runs.
        """
        if self.input_size is not None:
            self.W = 0.01 * self.rng.standard_normal((self.output_size, self.input_size))
        # Note: input_size stays the same, only weights are reset

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

    def step(self, X):
        X = np.asarray(X, dtype=float).reshape(-1)                       # (n)
        Y = self.forward(X)                                  # (k,)
        # ΔW = eta * ( y x^T - y y^T w )                   # yx^T is (k, n); y y^T W is (k, n)
        self.W += self.learning_rate * (np.outer(Y, X) - np.outer(Y, Y) @ self.W)
        # ΔW = eta * ( y x^T - y^2 w )
        # y_cov = np.outer(Y, Y)
        # y_var = np.diag(y_cov).copy()
        # self.W += self.learning_rate * (np.outer(Y, X) - y_var[None, :] @ self.W)
        return Y
    

    

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

    def step(self, X):
        X = np.asarray(X, dtype=float).reshape(-1)
        Y = self.forward(X)
        # ΔW = eta * (yx^T - BW), B is the lower triangular matrix of y y^T
        B = np.tril(np.outer(Y, Y))
        self.W += self.learning_rate * (np.outer(Y,X) - B @ self.W)
        return Y

class LeenMinimalPCA:
    """
    Activity-dependent Leen/Földiak PCA learner (NumPy).

    y = (I - V)^{-1} W x        # complete mode
    or y = W x                  # minimal mode (fast, robust for learning)

    Lateral (off-diagonal only):
        V_{ij} <- V_{ij} + η_v * (λ_i + λ_j) * ( - V_{ij} - C * <y_i y_j> )

    Forward (per unit i):
        w_i <- w_i + η_W * ( < x * s_i > - < y_i^2 > w_i )
        where s = y + V y  (vector per sample); i.e. s = (I + V) y

    Parameters
    ----------
    input_dim : int
    output_dim : int
    eta_w : float            # forward learning rate
    eta_v : float            # lateral learning rate (make this larger than eta_w)
    C : float                # coupling; pick > 1 (e.g., 1.5)
    ema_alpha : float        # EMA step for activity λ_i ≈ E[y_i^2]
    forward_mode : str       # "minimal", "complete", or "complete_first_order"
    symmetrize_V : bool      # keep V explicitly symmetric + zero diagonal after each step
    clip_V_spectral : float or None  # if not None, shrink V to keep ||V||_2 <= this value (e.g., 0.95)
    seed : int or None
    """
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 eta_w: float = 1e-3,
                 eta_v: float = 1e-2,
                 C: float = 1.5,
                 ema_alpha: float = 0.05,
                 symmetrize_V: bool = True,
                 seed: int | None = 0, 
                 clip_V_spectral: float = 0.95):

        self.input_size = input_size
        self.output_size = output_size
        self.eta_w = eta_w
        self.eta_v = eta_v
        self.C = C
        self.ema_alpha = ema_alpha
        self.symmetrize_V = symmetrize_V
        self.clip_V_spectral = clip_V_spectral

        self.rng = np.random.default_rng(seed)
        # Row-wise unit-norm init for W
        W = self.rng.normal(size=(self.output_size, self.input_size))
        W /= np.linalg.norm(W, axis=1, keepdims=True) + 1e-12
        self.W = W
        # Symmetric zero-diagonal init for q
        self.V = np.zeros((self.output_size, self.output_size))
        np.fill_diagonal(self.V, 0.0)
        # EMA of activities (start small positive to avoid zero)
        self.lam = np.full(self.output_size, 1e-6)

    # ---------- helpers ----------

    def _forward(self, X: np.ndarray) -> np.ndarray:
        """
        X: (T, input_size) -> Y: (T, output_size)
        """
        Y = X @ self.W.T  # (T, output_size) = W x
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
        if smax > self.clip_v_spectral:
            s = s * (self.clip_V_spectral / (smax + 1e-12))
            self.V = (u * s) @ vt
            self._symmetrize_V()    


    # ---------- public API ----------

    def step(self, X: np.ndarray):
        """
        One learning step on a batch X: shape (T, d).
        """
        X = np.asarray(X, dtype=float)
        T = X.shape[0]

        # Forward pass
        Y = self._forward(X)                                     # (T, m)

        # Batch moments
        y_cov = (Y.T @ Y) / T                                    # (output_size, output_size)  ~ < y_i y_j >
        y_var = np.diag(y_cov).copy()                            # (output_size,)

        # EMA update of activities λ_i ~ E[y_i^2]
        self.lam = (1.0 - self.ema_alpha) * self.lam + self.ema_alpha * y_var

        # ----- Lateral update (off-diagonal) -----
        lam_sum = self.lam[:, None] + self.lam[None, :]          # (output_size, output_size)  (λ_i + λ_j)
        # dq = self.eta_q * lam_sum * ( - self.q - self.C * y_cov )  # 09222025: This is wrong. Fixed below:
        dV = self.eta_v * (- lam_sum * self.V  - self.C * y_cov)             # decay term
        np.fill_diagonal(dV, 0.0)
        self.V += dV

        if self.symmetrize_V:
            self._symmetrize_V()
        self._clip_V_spectral_norm()
        # ----- Forward update (Hebb-Oja, gated by q) -----
        # s = (I + V) y  (per sample); with row-vectors: S = Y @ (I+V)^T
        S = Y @ (np.eye(self.output_size) + self.V).T                      # (T, output_size)
        # < x * s_i > as a matrix: (d, m) then transpose to (m, d)
        XS = (X.T @ S) / T                                       # (input_size, output_size)
        dW = self.eta_w * (XS.T - y_var[:, None] * self.W)       # (output_size, input_size)
        self.W += dW

        # Mild row renorm to avoid drift (Oja already stabilizes norms)
        row_norms = np.linalg.norm(self.W, axis=1, keepdims=True) + 1e-12
        self.W /= row_norms

    def transform(self, X: np.ndarray) -> np.ndarray:
        "Project X onto learned components (uses the current forward mode)."
        return self._forward(np.asarray(X, dtype=float))

    def reset(self):
        """
        Re-initialize W, V, and λ in the same style as __init__,
        but keep the RNG state (so repeated calls are not identical).
        """
        self.W = self.rng.normal(size=(self.output_size, self.input_size))
        self.W /= np.linalg.norm(self.W, axis=1, keepdims=True) + 1e-12
        self.V = np.zeros((self.output_size, self.output_size))
        np.fill_diagonal(self.V, 0.0)
        # self.V = self.rng.normal(size=(self.output_size, self.output_size))
        self.lam = np.full(self.output_size, 1e-6)

    @property
    def components_(self) -> np.ndarray:
        "Rows of W are the learned components (unit-norm). Shape: (output_size, input_size)"
        return self.W.copy()





### --------- Hebbian models with lateral connections --------- ###

class LeenCompletePCA:
    """
    Implementation of Leen's "complete" PCA model (Section 4).

    In this model, the neuron's output `y` is calculated recurrently,
    meaning the lateral connections `q` (eta in the paper) directly
    influence the cell's response in a feedback loop.
 
    Equations of Motion (ensemble average form):
    -------------------------------------------
    Forward Pass (Eq. 33):
        y = W x + q y
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
                 input_size: int, output_size: int,
                 eta_w: float = 1e-3,
                 eta_v: float = 1e-2,
                 C: float = 1.5,
                 ema_alpha: float = 0.05,
                 symmetrize_V: bool = True,
                 seed: int | None = 0, 
                 settling_steps: int = 10, 
                 noise_level: float = 0.01, 
                 clip_V_spectral: float = 0.95):   # <-- NEW
        self.input_size = input_size
        self.output_size = output_size
        self.eta_w = eta_w
        self.eta_v = eta_v
        self.C = C
        self.ema_alpha = ema_alpha
        self.symmetrize_V = symmetrize_V
        self.settling_steps = settling_steps
        self.noise_level = noise_level
        self.clip_V_spectral = clip_V_spectral

        self.rng = np.random.default_rng(seed)
        W = self.rng.normal(size=(self.output_size, self.input_size))
        W /= np.linalg.norm(W, axis=1, keepdims=True) + 1e-12 
        self.W = W
        self.initial_W = W.copy()
        # Symmetric zero-diagonal init for V
        # self.V = np.zeros((self.output_size, self.output_size))
        self.V = self.rng.normal(size=(self.output_size, self.output_size))
        # EMA of activities (start small positive to avoid zero)
        self.lam = np.full(self.output_size, 1e-6)
        # initialize the y_0. with shape (batch_size, m)

    # ---------- helpers ----------
    # ---------- NEW: reset ----------
    def reset(self):
        W = self.rng.normal(size=(self.output_size, self.input_size))
        W /= np.linalg.norm(W, axis=1, keepdims=True) + 1e-12
        self.W = W
        self.initial_W = W.copy()

        # Reinitialize lateral q and EMA lam exactly as in __init__
        # self.V = np.zeros((self.output_size, self.output_size))
        self.V = self.rng.normal(size=(self.output_size, self.output_size))
        self.lam = np.full(self.output_size, 1e-6)

    # ---------- modified forward function to Y = Wx + Qy ----------
    def _forward(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates the cell response by iteratively simulating the fast dynamics.
        y = W x + q y
        """
        # Line 1: Calculate the constant feed-forward input drive, Z = Wx
        # This is the external input to the recurrent system.
        Z = X @ self.W.T
        # Line 2: Initialize the neuron activities (y_0) to zero at the start.
        # Before the input arrives, the neurons are assumed to be silent. 
        # Y = self.rng.normal(scale=self.noise_level, size=Z.shape)
        Y = np.zeros_like(Z)
        # Line 3: Loop for a fixed number of steps to simulate settling.
        # This explicitly models the passage of time on the fast (millisecond) scale.
        
        for _ in range(self.settling_steps):
            # Line 4: Calculate the next state of neural activity.
            # Y = Z (feed-forward) + Y @ q.T (recurrent feedback)
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
        One learning step on a batch X: shape (T, d).

        """
        X = np.asarray(X, dtype=float)
        T = X.shape[0]

        # Forward pass (recurrent calculation)
        Y = self._forward(X)  # (T, m)
        # Batch moments from the complete response
        y_cov = (Y.T @ Y) / T      # (output_size, output_size) ~ <y y^T>
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
        hebb_term = (Y.T @ X) / T  # (m, d)
        oja_term = y_var[:, None] * self.W      # (output_size, input_size)
        dW = self.eta_w * (hebb_term - oja_term)
        self.W += dW


    def transform(self, X: np.ndarray) -> np.ndarray:
        "Project X onto learned components using the recurrent dynamics."
        return self._forward(np.asarray(X, dtype=float))

    @property
    def initial_weights_(self) -> np.ndarray:
        "Initial weights W, make sure the model is re-initialized. Shape: (m, d)"
        return self.initial_W


class OneShotLeenCompletePCA:
    """
    One-shot variant of Leen's "complete" PCA model.

    Instead of recurrent settling y_{t+1} = Wx + V y_t, this model
    uses a single lateral pass on the feedforward output:

        Y' = W X                      (feedforward drive)
        Y  = Y' + V Y' = (I + V) W X  (one-shot lateral interaction)

    The effective linear mapping is:

        A = (I + V) W
        Y = A X

    Learning rules follow the same *form* as LeenCompletePCA:

      - Lateral (activity-dependent anti-Hebbian):
            ΔV_ij = η_v * ( (λ_i + λ_j) * V_ij - C * <y_i y_j> ), i ≠ j
        where λ_i is an EMA of y_i^2.

      - Feedforward (Hebb–Oja on the complete response Y):
            ΔW = η_w * ( <Y X^T> - Diag(<y y^T>) W )

    Parameters
    ----------
    input_size : int
    output_size : int
    eta_w : float
        Learning rate for feedforward weights W.
    eta_v : float
        Learning rate for lateral connections V (typically > eta_w).
    C : float
        Coupling constant in the V update (e.g. > 1).
    ema_alpha : float
        EMA step for λ_i ≈ E[y_i^2].
    symmetrize_V : bool
        If True, keep V symmetric with zero diagonal after each step.
    seed : int or None
        Random seed for weight initialization.
    settling_steps, noise_level, clip_V_spectral : kept for API
        compatibility with LeenCompletePCA. settling_steps and
        noise_level are not used in the one-shot forward dynamics,
        but clip_V_spectral is still used to stabilize V.
    """
    def __init__(self,
                 input_size: int, output_size: int,
                 eta_w: float = 1e-3,
                 eta_v: float = 1e-2,
                 C: float = 1.5,
                 ema_alpha: float = 0.05,
                 symmetrize_V: bool = True,
                 seed: int | None = 0,
                 clip_V_spectral: float | None = 0.95):
        self.input_size = input_size
        self.output_size = output_size
        self.eta_w = eta_w
        self.eta_v = eta_v
        self.C = C
        self.ema_alpha = ema_alpha
        self.symmetrize_V = symmetrize_V
        self.clip_V_spectral = clip_V_spectral

        self.rng = np.random.default_rng(seed)

        # Initialize W with unit-norm rows
        W = self.rng.normal(size=(self.output_size, self.input_size))
        W /= np.linalg.norm(W, axis=1, keepdims=True) + 1e-12
        self.W = W
        self.initial_W = W.copy()

        # Symmetric zero-diagonal V
        # self.V = np.zeros((self.output_size, self.output_size))
        self.V = self.rng.normal(size=(self.output_size, self.output_size))
        # EMA of activities λ_i ≈ E[y_i^2]
        self.lam = np.full(self.output_size, 1e-6)

    # ---------- helpers ----------

    def reset(self):
        """
        Re-initialize W, V, and λ in the same style as __init__,
        but keep the RNG state (so repeated calls are not identical).
        """
        W = self.rng.normal(size=(self.output_size, self.input_size))
        W /= np.linalg.norm(W, axis=1, keepdims=True) + 1e-12
        self.W = W
        self.initial_W = W.copy()

        # self.V = np.zeros((self.output_size, self.output_size))
        self.V = self.rng.normal(size=(self.output_size, self.output_size))
        self.lam = np.full(self.output_size, 1e-6)

    def _forward(self, X: np.ndarray) -> np.ndarray:
        """
        One-shot forward pass:

            Y_ff = X W^T                 # feedforward
            Y    = Y_ff + Y_ff V^T       # lateral on Y_ff only

        X: (T, d)
        Returns Y: (T, m)
        """
        X = np.asarray(X, dtype=float)
        # Feedforward drive Y'
        Y_ff = X @ self.W.T        # (T, m); W is (m, d)
        # One-shot lateral on Y'
        Y = Y_ff + Y_ff @ self.V.T  # (T, m); V is (m, m)
        return Y

    def _symmetrize_V(self):
        self.V = 0.5 * (self.V + self.V.T)
        np.fill_diagonal(self.V, 0.0)

    def _clip_V_spectral_norm(self):
        if self.clip_V_spectral is None:
            return
        u, s, vt = np.linalg.svd(self.V, full_matrices=False)
        smax = s[0]
        if smax > self.clip_V_spectral:
            s = s * (self.clip_V_spectral / (smax + 1e-12))
            self.V = (u * s) @ vt
            if self.symmetrize_V:
                self._symmetrize_V()

    # ---------- public API ----------

    def step(self, X: np.ndarray):
        """
        One learning step on a batch X: shape (B, d).

        Uses:
          - One-shot forward Y = (I + V) W X
          - Activity-dependent anti-Hebbian update for V
          - Hebb–Oja update for W using Y
        """
        X = np.asarray(X, dtype=float)
        B = X.shape[0]

        # Forward pass (one-shot)
        Y = self._forward(X)  # (B, m)

        # Batch moments
        y_cov = (Y.T @ Y) / B          # (output_size, output_size) ≈ <y y^T>
        y_var = np.diag(y_cov).copy()  # (m,)   ≈ <y_i^2>

        # EMA of λ_i ≈ E[y_i^2]
        self.lam = (1.0 - self.ema_alpha) * self.lam + self.ema_alpha * y_var

        # ----- Lateral update: activity-dependent anti-Hebbian -----
        lam_sum = self.lam[:, None] + self.lam[None, :]  # (output_size, output_size)
        # ΔV_ij = η_v * ( (λ_i + λ_j) * V_ij - C * <y_i y_j> )


        # dV = self.eta_v * (lam_sum * self.V - self.C * y_cov) 
        dV = self.eta_v * (lam_sum * self.V - self.C * y_cov)

        ### 11222025: This is to TEST the lateral inhibition of the model. 
        # dV = np.minimum(dV, 0.0)
        np.fill_diagonal(dV, 0.0)
        self.V += dV

        if self.symmetrize_V:
            self._symmetrize_V()
        self._clip_V_spectral_norm()

        # ----- Feedforward update: Hebb–Oja on Y -----
        # Hebbian term <Y X^T>
        hebb_term = (Y.T @ X) / B      # (output_size, input_size)
        # Oja decay term Diag(<y y^T>) W
        oja_term = y_var[:, None] * self.W
        dW = self.eta_w * (hebb_term - oja_term)
        self.W += dW

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project X onto learned components with the one-shot mapping.
        """
        return self._forward(np.asarray(X, dtype=float))

    @property
    def components_(self) -> np.ndarray:
        """
        Rows of W are the learned components. Shape: (output_size, input_size)

        Return unit-norm rows for convenience.
        """
        W_norm = self.W.copy()
        W_norm /= np.linalg.norm(W_norm, axis=1, keepdims=True) + 1e-12
        return W_norm

    @property
    def initial_weights_(self) -> np.ndarray:
        """
        Initial weights W (after last reset). Shape: (output_size, input_size)
        """
        return self.initial_W


class OneShotLeenFreislebenPCA:
    """
    One-shot variant of the Freisleben (1993) parallel PCA algorithm.

    Architecture:
    -------------
    - Input layer:  d = input_size
    - Output layer: m = output_size
    - Feedforward weights: W ∈ R^{m x d}
    - Lateral connections: V ∈ R^{m x m}, symmetric, zero diagonal

    One-shot forward mapping:
    -------------------------
        Y_ff = W X                (feedforward)
        Y    = Y_ff + V Y_ff      (lateral on feedforward only)

    In matrix form (for each sample x):
        y = (I + V) W x

    Learning rules (Freisleben "new proposal", adapted to one-shot Y):
    ----------------------------------------------------------------
    1) Lateral update (Eq. 10 style):
           ΔV = -η_v ( V + <y y^T> )
       with diagonal forced to zero and optional symmetrization/clipping.

    2) Feedforward update (Eq. 11 style, Freisleben "new proposal"):
       Define modified output:
           ỹ = y + C * (V y)        # in matrix form: Y_tilde = Y + C * (Y V^T)

       Then batch updates:
           y_cov = <y y^T> ≈ (Y^T Y) / n_samples
           y_var = diag(y_cov)

           hebb_term = < ỹ x^T > ≈ (Y_tilde^T X) / n_samples
           oja_term  = y_var[:, None] * W

           ΔW = η_w * (hebb_term - oja_term)

    Optional:
        explicit_W_norm : if True, renormalize each row of W after each step.
    """

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 eta_w: float = 1e-3,
                 eta_v: float = 5e-3,
                 C: float = 3.0,
                 symmetrize_V: bool = True,
                 clip_V_spectral: float | None = 0.95,
                 seed: int | None = 0):
        self.input_size = input_size
        self.output_size = output_size
        self.eta_w = eta_w
        self.eta_v = eta_v
        self.C = C
        self.symmetrize_V = symmetrize_V
        self.clip_V_spectral = clip_V_spectral

        self.rng = np.random.default_rng(seed)

        # Initialize W: random unit-length rows
        W = self.rng.normal(size=(self.output_size, self.input_size))
        W /= np.linalg.norm(W, axis=1, keepdims=True) + 1e-12
        self.W = W
        self.initial_W = W.copy()

        # Symmetric zero-diagonal V
        # self.V = np.zeros((self.output_size, self.output_size), dtype=float)
        self.V = self.rng.normal(size=(self.output_size, self.output_size))

    # ---------- helpers ----------

    def reset(self):
        """
        Reinitialize W and V (same style as __init__), but keep RNG state.
        """
        W = self.rng.normal(size=(self.output_size, self.input_size))
        W /= np.linalg.norm(W, axis=1, keepdims=True) + 1e-12
        self.W = W
        self.initial_W = W.copy()

        # self.V = np.zeros((self.output_size, self.output_size), dtype=float)
        self.V = self.rng.normal(size=(self.output_size, self.output_size))

    def _forward(self, X: np.ndarray) -> np.ndarray:
        """
        One-shot forward pass:

            Y_ff = X W^T            # feedforward
            Y    = Y_ff + Y_ff V^T  # lateral on feedforward only

        X: (n_samples, d)
        Returns Y: (n_samples, output_size)
        """
        X = np.asarray(X, dtype=float)
        Y_ff = X @ self.W.T         # (n_samples, output_size)
        Y = Y_ff + Y_ff @ self.V.T  # (n_samples, output_size)
        return Y

    def _symmetrize_V(self):
        self.V = 0.5 * (self.V + self.V.T)
        np.fill_diagonal(self.V, 0.0)

    def _clip_V_spectral_norm(self):
        if self.clip_V_spectral is None:
            return
        u, s, vt = np.linalg.svd(self.V, full_matrices=False)
        smax = s[0]
        if smax > self.clip_V_spectral:
            s = s * (self.clip_V_spectral / (smax + 1e-12))
            self.V = (u * s) @ vt
            if self.symmetrize_V:
                self._symmetrize_V()

    # ---------- public API ----------

    def step(self, X: np.ndarray):
        """
        One learning step on a batch X: shape (n_samples, d).

        1) Compute one-shot Y = (I + V) W X.
        2) Update V with ΔV = -η_v ( V + <y y^T> ).
        3) Compute modified output Y_tilde = Y + C * (Y V^T).
        4) Update W with Hebb–Oja using Y_tilde and y_var.
        """
        X = np.asarray(X, dtype=float)
        T = X.shape[0]

        # ----- 1) One-shot forward pass -----
        Y = self._forward(X)        # (n_samples, output_size)

        # Batch covariance <y y^T>
        y_cov = (Y.T @ Y) / T       # (output_size, output_size)
        y_var = np.diag(y_cov).copy()  # (output_size,)

        # ----- 2) Lateral update (Reisleben eq. 10 style) -----
        dV = -self.eta_v * (self.V + y_cov)
        np.fill_diagonal(dV, 0.0)
        self.V += dV

        if self.symmetrize_V:
            self._symmetrize_V()
        self._clip_V_spectral_norm()

        # ----- 3) Modified output: ỹ = y + C * (V y) -----
        # In batch/matrix form: y_lat = Y V^T, then Y_tilde = Y + C * y_lat
        y_lat = Y @ self.V.T             # (T, output_size)
        Y_tilde = Y + self.C * y_lat     # (T, output_size)

        # ----- 4) Feedforward update (Hebb–Oja with Y_tilde) -----
        hebb_term = (Y_tilde.T @ X) / T  # (output_size, input_size)
        oja_term = y_var[:, None] * self.W # (output_size, input_size)

        dW = self.eta_w * (hebb_term - oja_term)
        self.W += dW

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project X using the one-shot parallel mapping.
        """
        return self._forward(np.asarray(X, dtype=float))

    @property
    def initial_weights_(self) -> np.ndarray:
        """
        Initial W at construction / last reset. Shape: (output_size, input_size)
        """
        return self.initial_W

class LeenCompletePCA_TD:
    """
    Implementation of Leen's "complete" PCA model (Section 4) with time-dependent (TD) weights.

    In this model, the neuron's output `y` is calculated recurrently,
    meaning the lateral connections `q` (eta in the paper) directly
    influence the cell's response in a feedback loop.
 
    Equations of Motion (ensemble average form):
    -------------------------------------------
    Forward Pass (Eq. 33):
        y_{t+1} = W(t) x_t + y_{t} V(t)^T
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
                 input_size: int, output_size: int,
                 eta_w: float = 1e-3,
                 eta_v: float = 1e-2,
                 C: float = 1.5,
                 ema_alpha: float = 0.05,
                 symmetrize_V: bool = True,
                 seed: int | None = 0, 
                 clip_V_spectral: float = 0.95):   # <-- NEW
        self.input_size = input_size
        self.output_size = output_size
        self.eta_w = eta_w
        self.eta_v = eta_v
        self.C = C
        self.ema_alpha = ema_alpha
        self.symmetrize_V = symmetrize_V
        self.clip_V_spectral = clip_V_spectral
        self.Y = np.zeros((self.output_size))

        self.rng = np.random.default_rng(seed)
        W = self.rng.normal(size=(self.output_size, self.input_size))
        W /= np.linalg.norm(W, axis=1, keepdims=True) + 1e-12 
        self.W = W
        self.initial_W = W.copy()
        # Symmetric zero-diagonal init for V
        # self.V = np.zeros((self.output_size, self.output_size))
        self.V = self.rng.normal(size=(self.output_size, self.output_size))
        # EMA of activities (start small positive to avoid zero)
        self.lam = np.full(self.output_size, 1e-6)
        # initialize the y_0. with shape (batch_size, m)

    # ---------- helpers ----------
    # ---------- NEW: reset ----------
    def reset(self):
        W = self.rng.normal(size=(self.output_size, self.input_size))
        W /= np.linalg.norm(W, axis=1, keepdims=True) + 1e-12
        self.W = W
        self.initial_W = W.copy()

        # Reinitialize lateral q and EMA lam exactly as in __init__
        # self.V = np.zeros((self.output_size, self.output_size))
        self.V = self.rng.normal(size=(self.output_size, self.output_size))
        self.lam = np.full(self.output_size, 1e-6)

    # ---------- modified forward function to Y_{t+1} = Wx + y_{t} V^T ----------
    def _forward(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates the cell response by iteratively simulating the fast dynamics.
        $$ y_{t+1} = W x_t + V(t) y_t $$
        """

        Z = X @ self.W.T # (1, N) @ (N, m) = (1, m)
        _Y = self.Y @ self.V.T # (1, m) @ (m, m) = (1, m)
        Y = Z + _Y # (1, m) -> (1, m)
        return Y # (1, m)
    
    def _forward_transform(self, X: np.ndarray) -> np.ndarray:
        Y = X @ self.W.T + self.Y @ self.V.T
        return Y
    # def _forward(self, X: np.ndarray) -> np.ndarray:
    #     """"
    #     Calculates the cell response by iteratively simulating the fast dynamics.
    #     y = W x + q y
    #     """
    #     # Line 1: Calculate the constant feed-forward input drive, Z = Wx
    #     # This is the external input to the recurrent system.
    #     Z = X @ self.W.T
    #     # Line 2: Initialize the neuron activities (y_0) to zero at the start.
    #     # Before the input arrives, the neurons are assumed to be silent. 
    #     Y = self.rng.normal(scale=self.noise_level, size=Z.shape)
    #     # Y = np.zeros_like(Z)
    #     # Line 3: Loop for a fixed number of steps to simulate settling.
    #     # This explicitly models the passage of time on the fast (millisecond) scale.
        
    #     for _ in range(self.settling_steps):
    #         # Line 4: Calculate the next state of neural activity.
    #         # Y = Z (feed-forward) + Y @ q.T (recurrent feedback)
    #         # The current activity Y is fed back through the lateral connections q
    #         # and added to the constant external drive Z.
    #         Y = Z + Y @ self.V.T
    #     # Line 5: Return the final, settled activity after the loop.
    #     return Y

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
        One learning step on a time series X: shape (1, N).
        """
        X = np.asarray(X, dtype=float)
        T = X.shape[0]

        # Forward pass (recurrent calculation)
        self.Y = self._forward(X)  # (1, m)
        # Batch moments from the complete response
        y_cov = (self.Y.T @ self.Y) / T      # (output_size, output_size) ~ <y y^T>
        y_var = np.diag(y_cov).copy()  # (m,)   ~ <y_i^2>

        # EMA update of activities λ_i ~ E[y_i^2]
        self.lam = (1.0 - self.ema_alpha) * self.lam + self.ema_alpha * y_var

        # ----- Lateral update (Activity-Dependent Anti-Hebbian) -----
        lam_sum = self.lam[:, None] + self.lam[None, :]  # (output_size, output_size) (λ_i + λ_j)
        # Implements Δq_ij = η_q * ( (λ_i + λ_j) * q_ij - C * <y_i y_j> )
        dV = self.eta_v * (lam_sum * self.V - self.C * y_cov)
        np.fill_diagonal(dV, 0.0)
        self.V += dV

        if self.symmetrize_V:
            self._symmetrize_V()
        self._clip_V_spectral_norm()

        # ----- Forward update (Hebb-Oja on complete response) -----
        # Implements Δw = η_w * ( <y x^T> - Diag(<y y^T>) w )
        hebb_term = (self.Y.T @ X) / T  # (m, 1) @ (1, N) = (m, N)
        oja_term = y_var[:, None] * self.W      # (output_size, input_size)
        dW = self.eta_w * (hebb_term - oja_term)
        self.W += dW

    def transform(self, X: np.ndarray) -> np.ndarray:
        "Project X onto learned components using the recurrent dynamics."
        return self._forward_transform(np.asarray(X, dtype=float))

    @property
    def initial_weights_(self) -> np.ndarray:
        "Initial weights W, make sure the model is re-initialized. Shape: (output_size, input_size)"
        return self.initial_W



class FoldiakPCA:
    """
    A two-layer network with a Hebbian layer and an Anti-Hebbian layer.
    """

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 eta_w: float = 1e-3,
                 eta_v: float = 5e-3,
                 symmetrize_V: bool = True,
                 clip_V_spectral: float | None = 0.95,
                 seed: int | None = 0):
        self.input_size = input_size
        self.output_size = output_size
        self.eta_w = eta_w
        self.eta_v = eta_v
        self.symmetrize_V = symmetrize_V
        self.clip_V_spectral = clip_V_spectral

        self.rng = np.random.default_rng(seed)

        # Initialize W: random unit-length rows
        W = self.rng.normal(size=(self.output_size, self.input_size))
        W /= np.linalg.norm(W, axis=1, keepdims=True) + 1e-12
        self.W = W
        self.initial_W = W.copy()

        # Symmetric zero-diagonal V
        # self.V = np.zeros((self.output_size, self.output_size), dtype=float)
        self.V = self.rng.normal(size=(self.output_size, self.output_size))

    # ---------- helpers ----------

    def reset(self):
        """
        Reinitialize W and V (same style as __init__), but keep RNG state.
        """
        W = self.rng.normal(size=(self.output_size, self.input_size))
        W /= np.linalg.norm(W, axis=1, keepdims=True) + 1e-12
        self.W = W
        self.initial_W = W.copy()

        # self.V = np.zeros((self.output_size, self.output_size), dtype=float)
        self.V = self.rng.normal(size=(self.output_size, self.output_size))

    def _forward(self, X: np.ndarray) -> np.ndarray:
        """
        One-shot forward pass:

            Y_ff = X W^T            # feedforward
            Y    = Y_ff + Y_ff V^T  # lateral on feedforward only

        X: (n_samples, d)
        Returns Y: (n_samples, output_size)
        """
        X = np.asarray(X, dtype=float)
        Y_ff = X @ self.W.T         # (n_samples, output_size)
        Y = Y_ff + Y_ff @ self.V.T  # (n_samples, output_size)
        return Y

    def _symmetrize_V(self):
        self.V = 0.5 * (self.V + self.V.T)
        np.fill_diagonal(self.V, 0.0)

    def _clip_V_spectral_norm(self):
        if self.clip_V_spectral is None:
            return
        u, s, vt = np.linalg.svd(self.V, full_matrices=False)
        smax = s[0]
        if smax > self.clip_V_spectral:
            s = s * (self.clip_V_spectral / (smax + 1e-12))
            self.V = (u * s) @ vt
            if self.symmetrize_V:
                self._symmetrize_V()

    # ---------- public API ----------

    def step(self, X: np.ndarray):
        """
        One learning step on a batch X: shape (B, d).

        Uses:
          - One-shot forward Y = (I + V) W X
          - Activity-dependent anti-Hebbian update for V
          - Hebb–Oja update for W using Y
        """
        X = np.asarray(X, dtype=float)
        T = X.shape[0]

        # Forward pass (one-shot)
        Y = self._forward(X)  # (B, m)

        # Batch moments
        y_cov = (Y.T @ Y) / T          # (output_size, output_size) ≈ <y y^T>
        y_var = np.diag(y_cov).copy()  # (m,)   ≈ <y_i^2>

        # ----- Lateral update: activity-dependent anti-Hebbian -----
        dV = self.eta_v * (- y_cov)
        np.fill_diagonal(dV, 0.0)
        self.V += dV

        if self.symmetrize_V:
            self._symmetrize_V()
        self._clip_V_spectral_norm()

        # ----- Feedforward update: Hebb–Oja on Y -----
        # Hebbian term <Y X^T>
        hebb_term = (Y.T @ X) / T      # (output_size, input_size)
        # Oja decay term Diag(<y y^T>) W
        oja_term = y_var[:, None] * self.W
        dW = self.eta_w * (hebb_term - oja_term)
        self.W += dW

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project X using the one-shot parallel mapping.
        """
        return self._forward(np.asarray(X, dtype=float))

    @property
    def components_(self) -> np.ndarray:
        """
        Rows of W are the learned components. Shape: (output_size, input_size).

        Return unit-norm rows for convenience.
        """
        W_norm = self.W.copy()
        W_norm /= np.linalg.norm(W_norm, axis=1, keepdims=True) + 1e-12
        return W_norm

    @property
    def initial_weights_(self) -> np.ndarray:
        """
        Initial W at construction / last reset. Shape: (output_size, input_size)
        """
        return self.initial_W