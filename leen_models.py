import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import hebbian.utils as utils
import pandas as pd

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




# --- PCA alignment helpers ---
def best_match_alignment(W_rows, PC_rows):
    """
    W_rows: (m, d) learned components, row-normalized
    PC_rows: (m, d) true PCs, row-normalized in descending variance order
    Returns:
      perm: indices of PCs assigned to each W row
      corrs: absolute correlations after optimal assignment
      C: full |cosine| matrix (m x m)
    """
    # normalize
    Wn = W_rows / (np.linalg.norm(W_rows, axis=1, keepdims=True) + 1e-12)
    PCn = PC_rows / (np.linalg.norm(PC_rows, axis=1, keepdims=True) + 1e-12)
    # cosine matrix
    C = np.abs(Wn @ PCn.T)

        # Hungarian solves a min-cost problem; convert to cost = 1 - C
    r, c = linear_sum_assignment(1.0 - C)

    return c, C[np.arange(C.shape[0]), c], C

def best_match_align_timeseries(Y, PCs, metric="corr", absolute=True, return_aligned=False):
    """
    Match columns of Y to columns of PCs by maximizing pairwise similarity.

    Parameters
    ----------
    Y : array-like, shape (n_samples, m)
        Model outputs; each column is a component across samples.
    PCs : array-like, shape (n_samples, m)
        Principal component *scores* (or any reference components), column-wise.
    metric : {"corr", "cosine"}, default "corr"
        Similarity to maximize. "corr" = Pearson correlation (column-wise, mean-centered);
        "cosine" = cosine similarity (L2-normalized columns).
    absolute : bool, default True
        If True, ignore sign (common with PCA). If False, keep signed similarity.
    return_aligned : bool, default False
        If True, also return (Y_aligned, PCs_perm) where Y columns are sign-flipped
        and PCs are permuted to the best match order.

    Returns
    -------
    perm : ndarray, shape (m,)
        For each Y column i, PC column index perm[i] is the best match.
    sims : ndarray, shape (m,)
        Similarity values for those matches (abs or signed per `absolute`).
    S : ndarray, shape (m, m)
        Full similarity matrix where S[i, j] compares Y[:, i] vs PCs[:, j].
    signs : ndarray, shape (m,)
        Orientation (+1/-1) to multiply Y[:, i] so it aligns with PCs[:, perm[i]].
    (optional) Y_aligned, PCs_perm
        Returned only if return_aligned=True.
    """
    Y = np.asarray(Y, dtype=float)
    PCs = np.asarray(PCs, dtype=float)
    if Y.shape != PCs.shape or Y.ndim != 2:
        raise ValueError("Y and PCs must have identical shape (n_samples, m).")

    n, m = Y.shape

    def _zscore(A):
        mu = A.mean(axis=0, keepdims=True)
        sd = A.std(axis=0, ddof=1, keepdims=True)
        sd = np.where(sd == 0.0, 1.0, sd)
        return (A - mu) / sd

    if metric == "corr":
        Yz, PCz = _zscore(Y), _zscore(PCs)
        # correlation matrix between columns
        S = (Yz.T @ PCz) / (n - 1)
    elif metric == "cosine":
        Yn = Y / (np.linalg.norm(Y, axis=0, keepdims=True) + 1e-12)
        PCn = PCs / (np.linalg.norm(PCs, axis=0, keepdims=True) + 1e-12)
        S = Yn.T @ PCn
    else:
        raise ValueError("metric must be 'corr' or 'cosine'.")

    S_use = np.abs(S) if absolute else S

    # Solve the max-sum assignment (Hungarian solves min-cost; convert similarity to cost)
    row_ind, col_ind = linear_sum_assignment(1.0 - S_use)
    perm = np.empty(m, dtype=int)
    perm[row_ind] = col_ind

    sims = S_use[np.arange(m), perm]
    signs = np.sign(S[np.arange(m), perm])  # orientation from *signed* S
    signs[signs == 0] = 1

    if return_aligned:
        Y_aligned = Y * signs  # broadcast flips per column
        PCs_perm = PCs[:, perm]
        return perm, sims, S, signs, Y_aligned, PCs_perm

    return perm, sims, S, signs


def true_pcs_rows(X, m):
    """Return top-m PCs as ROWS (shape m x d)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, VT = np.linalg.svd(Xc, full_matrices=False)
    return VT[:m, :]  # rows are PCs

def get_pc_scores(X, eigenvectors, m=None):
    Xc = X - X.mean(axis=0, keepdims=True)
    PC_scores = Xc @ eigenvectors
    if m is not None:
        PC_scores = PC_scores[:, :m]
    return PC_scores


def subspace_max_angle_deg(W_rows, PC_rows):
    """Largest principal angle between the two m-dim subspaces (in degrees)."""
    # Orthonormal bases for colspaces of W^T and PC^T
    Qw, _ = np.linalg.qr(W_rows.T)
    Qp, _ = np.linalg.qr(PC_rows.T)
    s = np.linalg.svd(Qw.T @ Qp, compute_uv=False)
    return float(np.degrees(np.arccos(np.clip(s.min(), -1.0, 1.0))))


def compute_eigen_decomposition(X, center_data=True, return_covariance=False):
    """
    Compute eigenvalues and eigenvectors from input data X.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data matrix
    center_data : bool, default=True
        Whether to center the data (subtract mean) before computing covariance
    return_covariance : bool, default=False
        Whether to also return the covariance matrix
        
    Returns
    -------
    eigenvalues : ndarray, shape (n_features,)
        Eigenvalues in descending order
    eigenvectors : ndarray, shape (n_features, n_features)
        Eigenvectors as columns (each column is an eigenvector)
    covariance : ndarray, shape (n_features, n_features), optional
        Covariance matrix (only returned if return_covariance=True)
    """
    X = np.asarray(X, dtype=float)
    
    # Center the data if requested
    if center_data:
        X_centered = X - X.mean(axis=0, keepdims=True)
    else:
        X_centered = X
    
    # Compute covariance matrix
    n_samples = X_centered.shape[0]
    covariance = (X_centered.T @ X_centered) / (n_samples - 1)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    
    # Sort in descending order (largest eigenvalues first)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    if return_covariance:
        return eigenvalues, eigenvectors, covariance
    else:
        return eigenvalues, eigenvectors

# ------ Data generation helpers ------

def generate_covariance_matrix(dim, method='random', eigenvalues=None, condition_number=None, seed=None):
    """
    Generate a symmetric positive-semidefinite covariance matrix.
    
    Parameters
    ----------
    dim : int
        Dimension of the covariance matrix (dim x dim)
    method : str, default='random'
        Method to generate the matrix:
        - 'random': Random positive definite matrix
        - 'eigenvalues': Use specified eigenvalues
        - 'condition': Use specified condition number
        - 'toeplitz': Toeplitz matrix (correlation decreases with distance)
        - 'block': Block diagonal structure
    eigenvalues : array-like, optional
        Specific eigenvalues to use (must be non-negative)
    condition_number : float, optional
        Condition number (max_eigenval / min_eigenval) for 'condition' method
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    cov : ndarray, shape (dim, dim)
        Symmetric positive-semidefinite covariance matrix
    """
    if seed is not None:
        np.random.seed(seed)
    
    if method == 'random':
        # Method 1: Random positive definite matrix
        A = np.random.randn(dim, dim)
        cov = A @ A.T  # This ensures positive semidefinite
        # Add diagonal to ensure positive definite
        cov += np.eye(dim) * 0.1
        
    elif method == 'eigenvalues':
        # Method 2: Use specified eigenvalues
        if eigenvalues is None:
            eigenvalues = np.linspace(1.0, 0.1, dim)
        eigenvalues = np.asarray(eigenvalues)
        assert len(eigenvalues) == dim, "Eigenvalues length must match dimension"
        assert np.all(eigenvalues >= 0), "All eigenvalues must be non-negative"
        
        # Generate random orthogonal matrix
        Q, _ = np.linalg.qr(np.random.randn(dim, dim))
        cov = Q @ np.diag(eigenvalues) @ Q.T
        
    elif method == 'condition':
        # Method 3: Use specified condition number
        if condition_number is None:
            condition_number = 10.0
        assert condition_number >= 1.0, "Condition number must be >= 1"
        
        # Generate eigenvalues with specified condition number
        max_eigenval = 1.0
        min_eigenval = max_eigenval / condition_number
        eigenvalues = np.linspace(max_eigenval, min_eigenval, dim)
        
        # Generate random orthogonal matrix
        Q, _ = np.linalg.qr(np.random.randn(dim, dim))
        cov = Q @ np.diag(eigenvalues) @ Q.T
        
    elif method == 'toeplitz':
        # Method 4: Toeplitz matrix (correlation structure)
        rho = 0.7  # correlation parameter
        cov = np.zeros((dim, dim))
        for i in range(dim):
            for j in range(dim):
                cov[i, j] = rho ** abs(i - j)
        # Scale to have unit diagonal
        cov = cov / np.diag(cov)[:, None]
        
    elif method == 'block':
        # Method 5: Block diagonal structure
        block_size = min(3, dim)
        cov = np.zeros((dim, dim))
        
        for i in range(0, dim, block_size):
            end_idx = min(i + block_size, dim)
            block_dim = end_idx - i
            if block_dim > 1:
                # Create a small positive definite block
                A = np.random.randn(block_dim, block_dim)
                block = A @ A.T + np.eye(block_dim) * 0.1
                cov[i:end_idx, i:end_idx] = block
            else:
                cov[i, i] = 1.0
                
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Ensure symmetry (should already be symmetric, but just in case)
    cov = (cov + cov.T) / 2
    
    return cov

def generate_samples(d, number_of_samples):
    mu = np.full(d, 0.0)
    cov = generate_covariance_matrix(d, method='random')
    samples = np.random.multivariate_normal(mu, cov, size=number_of_samples)
    samples -= samples.mean(axis=0, keepdims=True)
    return samples

#---------- Traing and Testing helper ------------------
def W_aligned_training_testing(model, X, batch_size=100):
    hist_best_corr = []
    number_of_samples = X.shape[0]
    for step in range(0, number_of_samples, batch_size):
        X_train = X[step:step+batch_size]
        model.step(X_train)
        # cosine-monitor every {batch_size} steps (optional)
        if step % batch_size == 0:
            W = model.components_
            PC = true_pcs_rows(X, model.m)
            ordered_PCs , best_corrs, _ = best_match_alignment(W, PC)
            hist_best_corr.append(best_corrs)
            print(f"step {step:4d} | order = {ordered_PCs} | best|corr|={np.round(best_corrs,3)}")

    hist_best_corr = np.array(hist_best_corr)
    display_dict = {
                f"neuron {i+1}": hist_best_corr[:, i] for i in range(model.m)
            }

    utils.visualize_alignment(list(display_dict.values()), list(display_dict.keys()), model.m, "batch")


def Y_aligned_training_testing(model, X, batch_size=1, alignment = "corr"):
    number_of_samples = X.shape[0]
    for step in range(0, number_of_samples, batch_size):
        X_train = X[step:step+batch_size]
        model.step(X_train)
    # post-training, re-compute outputs for all samples using the learned weights
    Y_output = model.transform(X)
    # get true PC scores
    PC_scores = get_pc_scores(X, true_pcs_rows(X, model.d), m=model.m)
    order, similarity, _, _ = best_match_align_timeseries(Y_output, PC_scores, metric=alignment)
    # print(f"order = {order} | best|corr|= {np.round(similarity,3)}")
    return order, similarity



# ---------- Online training testing ------------------
def Y_aligned_training_testing_online(model, X, batch_size=1, alignment = "corr", graph=True):
    number_of_samples = X.shape[0]
    similarity_list = []
    # get true PC scores
    PC_scores = get_pc_scores(X, true_pcs_rows(X, model.m))
    print(f'W: {model.W}; V: {model.V}')
    for step in range(0, number_of_samples, batch_size):
        X_train = X[step:step+batch_size]
        model.step(X_train)
        # use the X_mask to incrementally reveal the input matrix, and update the model
        Y_output = model.transform(X)
        order, similarity, _, _ = best_match_align_timeseries(Y_output, PC_scores, metric=alignment)
        similarity_list.append(similarity)
        if step % 1000 == 0:
            print(f"step {step:4d} | order = {order} | best|corr|= {np.round(similarity,3)}")
    similarity_list = np.array(similarity_list)
    if graph:
        utils.visualize_alignment([similarity_list[:, i] for i in range(model.m)],
                                    [f"Neuron {i+1}" for i in range(model.m)],
                                    model.m, "trials")
    return order, similarity


# ---------- Averaged training testing ------------------
def averaged_training_testing_W(model, model_paras, X, runs = 10, batch_size=100, graph=True):
    df_best_corr = pd.DataFrame(data=np.full((runs, model_paras['output_dim'] ), np.nan), 
                                columns=[f"PC_{i+1}" for i in range(model_paras['output_dim'])],
                                index=[f"run_{i+1}" for i in range(runs)])
    number_of_samples = X.shape[0]
    for run in range(runs):
        _model = model(**model_paras)  # re-initialize model for each run if needed
        for step in range(0, number_of_samples, batch_size):
            X_train = X[step:step+batch_size]
            _model.step(X_train)
            W = _model.components_
            PC = true_pcs_rows(X, _model.m)
            ordered_PCs, best_corrs, _ = best_match_alignment(W, PC)
        # based on the order_PCs, sort the best_corrs to match PC order
        combined_lists = zip(ordered_PCs, best_corrs)
        # Sort the pairs based on PC_orders
        sorted_combined_lists = sorted(combined_lists)
        # Unzip the sorted pairs
        _, sorted_best_corrs = zip(*sorted_combined_lists)
        df_best_corr.iloc[run, :] = sorted_best_corrs
        print(f"run {run+1:2d} | order = {ordered_PCs} | best|corr|={np.round(best_corrs,3)}")
    if graph:
        # plot the bar graph with error bars of mean and std based on df_best_corr
        means = df_best_corr.mean(axis=0)
        stds = df_best_corr.std(axis=0)
        plt.figure(figsize=(10, 6))
        plt.bar(means.index, means.values, yerr=stds.values, capsize=5)
        # plt.ylim(0, 1.1)
        plt.ylabel('Mean Absolute Correlation')
        plt.title('Mean Absolute Correlation with True PCs Across Runs')
        plt.show()
    return df_best_corr