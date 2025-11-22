# -- coding: utf-8 --
"""
Functions used to assist training and testing Hebbian models.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import pearsonr
from scipy.optimize import linear_sum_assignment
import pandas as pd
from matplotlib.lines import Line2D
#--------------------------------
# Statistical functions
#--------------------------------

def pca_topk(X, k, center=True):
    """
    PCA via SVD. Returns the top-k principal components.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data (each row is a sample).
    k : int
        Number of principal components to keep (1 <= k <= min
        (n_samples, n_features)).
    center : bool, default True
        If True, subtract column means before SVD.

    Returns
    -------
    pcs : ndarray, shape (n_features, k)
        Principal directions (each column is a PC vector).
    eigvals : ndarray, shape (k,)
        Eigenvalues (variances) along those PCs.
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    if not (1 <= k <= min(n, d)):
        raise ValueError(f"k must be in [1, {min(n, d)}], got {k}.")

    mean_ = X.mean(axis=0) if center else np.zeros(d)
    Xc = X - mean_ if center else X

    # SVD: Xc = U S Vt  (Vt rows are PCs)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    pcs = Vt[:k].T                    # (d, k)
    # Eigenvalues of covariance = S^2 / (n-1)
    eigvals_full = (S**2) / (n - 1 if n > 1 else 1)
    eigvals = eigvals_full[:k]
    # scores
    scores = Xc @ pcs

    return pcs, eigvals, scores

def random_vector_chance_prob(input_size):
    return np.sqrt(2/(np.pi * input_size))

#--------------------------------
# Training and testing functions
#--------------------------------

def cosine_alignment(u, v):
    """|cos(theta)| between two vectors u, v."""
    u = u / (np.linalg.norm(u) + 1e-12)
    v = v / (np.linalg.norm(v) + 1e-12)
    return float(abs(u @ v))

def correlation_alignment(u, v):
    r, _ = pearsonr(u, v)
    return float(abs(r))

def bar_graph_with_error_bars(df_best_corr, title):
    figure = plt.figure()
    means = df_best_corr.mean(axis=0)
    stds = df_best_corr.std(axis=0)
    plt.figure(figsize=(10, 6))
    plt.bar(means.index, means.values, yerr=stds.values, capsize=5)
    plt.ylabel('Mean Absolute Correlation')
    plt.title(title)
    plt.show()

def training_testing(model, data, m, runs =10, graph = True, alignment = "eigenvectors"):
    df_best_corr = pd.DataFrame(data=np.full((runs, m), np.nan), 
                            columns=[f"PC_{i+1}" for i in range(m)],
                            index=[f"run_{i+1}" for i in range(runs)])
    for run in range(runs):
        print(f"run {run+1:2d}")
        model.reset()
        order, similarity = Y_aligned_training_testing_online(model, X=data, graph=True, alignment=alignment)
        # print(f"After first phase: model.W = {model.W}, model.V: {model.V}")

        # print(f"Y_output_after_first_phase: {model.transform(data_before)}")
       # based on the order, sort the similarity to match PC order
        combined_lists = zip(order, similarity)
        # Sort the pairs based on PC_orders
        sorted_combined_lists = sorted(combined_lists)
        # Unzip the sorted pairs
        _, ordered_similarity = zip(*sorted_combined_lists)
        df_best_corr.iloc[run, :] = ordered_similarity 

    if graph == True:
        bar_graph_with_error_bars(df_best_corr, "Mean Absolute Correlation (before)")

# to test the eigenvalue
def estimate_output_variances(X, weights, center=True, unit_weights=True):
    """
    Compute Var(Y) with current W on a dataset X (rows=samples).
    If center=True, subtract column means of X first (recommended).
    If unit_weights=True, normalize columns of W before projecting.
    Returns: var_y (k,), the per-output variances.
    """
    X = np.asarray(X, float)
    if center:
        X = X - X.mean(axis=0, keepdims=True)
    W = weights.copy() #(k, n)
    if unit_weights:
        W /= (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)

    Y = W@X.T                   # (k, n) x (n, samples)
    return Y.var(axis=1, ddof=1)

#--------------------------------
# Visualization functions
#--------------------------------

def visualize_alignment(hist_list, label_list, input_size, xlabel = "epochs"):
    """
    This function takes the history of the previous weights from different models to visualize the 
    alignment between weights and eigenvector. 
    hist_list: list;
    label_list: list;
    """
    figure = plt.figure(figsize=(15, 5), dpi=100)
    for hist, label in zip(hist_list, label_list):
        plt.plot(hist, label=label)
    
    # plot the chance level fitting in random vector
    plt.axhline(random_vector_chance_prob(input_size=input_size), label="chance_level", color = 'black', linestyle="--")
    plt.xlabel(xlabel)
    plt.ylabel("Alignment to PC (|cos θ|)")
    plt.ylim(0, 1.01)
    plt.legend()
    plt.show()

def plot_samples_and_top2_pcs(X, center=True, scale=2.5, annotate=True, ax=None):
    """
    Visualize samples and the first two principal components (PCs).

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input samples (rows = samples).
    center : bool, default True
        Subtract column means before PCA.
    scale : float, default 2.5
        How long to draw PC arrows (in units of sqrt(eigenvalue)).
    annotate : bool, default True
        Whether to label the PC arrows with names and eigenvalues.
    ax : matplotlib Axes or None
        If provided, draw on this axes; otherwise create a new figure.

    Returns
    -------
    info : dict with keys:
        'pcs'     : (n_features, 2) first two principal directions (columns)
        'eigvals' : (2,) eigenvalues (variances) for PC1, PC2
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape

    mean = X.mean(axis=0) if center else np.zeros(d)
    Xc = X - mean if center else X

    # PCA via SVD
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    # PCs are columns of V = Vt.T; eigenvalues = S^2/(n-1)
    V = Vt.T
    eigvals_full = (S**2) / (n - 1 if n > 1 else 1)

    # order by descending variance
    order = np.argsort(eigvals_full)[::-1]
    V = V[:, order]
    eigvals = eigvals_full[order]

    # What to plot:
    if d == 2:
        # plot in original centered coordinates
        Xplot = Xc
        # PCs are already in data coordinates
        P = V[:, :2]                # (2,2)
        lam = eigvals[:2]
        origin = np.zeros(2)
        xlabel, ylabel = "x1 (centered)", "x2 (centered)"
    else:
        # project data to first two PCs (scores)
        P = V[:, :2]                # (d,2)
        lam = eigvals[:2]
        Xplot = Xc @ P              # (n,2)
        origin = np.zeros(2)
        xlabel, ylabel = "PC1 score", "PC2 score"

    # plotting
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(Xplot[:, 0], Xplot[:, 1], s=10, alpha=0.6)
    ax.axhline(0, lw=0.6)
    ax.axvline(0, lw=0.6)

    # draw arrows for PC1 and PC2
    # length ∝ sqrt(eigenvalue); scaled by 'scale'
    if d == 2:
        # arrows in original coordinates
        for i in range(2):
            vec = scale * np.sqrt(lam[i]) * P[:, i]
            ax.arrow(origin[0], origin[1], vec[0], vec[1],
                     length_includes_head=True, head_width=0.05*np.linalg.norm(vec),
                     linewidth=2)
            if annotate:
                ax.text(vec[0]*1.05, vec[1]*1.05, f"PC{i+1}\nλ={lam[i]:.3f}")
    else:
        # In projected space, PCs are coordinate axes; draw aligned arrows
        for i in range(2):
            vec = np.zeros(2)
            vec[i] = scale * np.sqrt(lam[i])
            ax.arrow(origin[0], origin[1], vec[0], vec[1],
                     length_includes_head=True, head_width=0.05*np.linalg.norm(vec),
                     linewidth=2)
            if annotate:
                ax.text(vec[0]*1.05, vec[1]*1.05, f"PC{i+1}\nλ={lam[i]:.3f}")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("Samples and top-2 PCs")

    return {"pcs": V[:, :2], "eigvals": eigvals[:2], "mean": mean}


def _set_axes_equal_3d(ax):
    """Make 3D axes have equal scale."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    x_mid = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_mid = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_mid = np.mean(z_limits)
    radius = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_mid - radius, x_mid + radius])
    ax.set_ylim3d([y_mid - radius, y_mid + radius])
    ax.set_zlim3d([z_mid - radius, z_mid + radius])

def plot_samples_and_top3_pcs_3d(X, center=True, scale=2.5, annotate=True, ax=None):
    """
    Visualize (n,3) samples and the top three principal components in 3D.

    Parameters
    ----------
    X : array-like, shape (n_samples, 3)
        Input samples (rows = samples).
    center : bool, default True
        Subtract column means before PCA.
    scale : float, default 2.5
        Scales arrow lengths; each arrow length is scale * sqrt(eigenvalue).
    annotate : bool, default True
        Label arrows with PC index and eigenvalue.
    ax : mpl_toolkits.mplot3d.Axes3D or None
        If provided, draw on this axes; otherwise create a new 3D figure.

    Returns
    -------
    info : dict with keys
        'pcs'     : (3, 3) principal directions as columns (PC1..PC3)
        'eigvals' : (3,) eigenvalues (variances) for PC1..PC3
        'mean'    : (3,) mean vector used for centering
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    assert d == 3, f"Expected X to have 3 features, got {d}"

    mean = X.mean(axis=0) if center else np.zeros(d)
    Xc = X - mean if center else X

    # PCA via SVD on centered data
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt.T                                 # PCs as columns
    eigvals_full = (S**2) / (n - 1 if n > 1 else 1)

    # sort by descending eigenvalue (usually already sorted, but be safe)
    order = np.argsort(eigvals_full)[::-1]
    V = V[:, order]
    eigvals = eigvals_full[order]

    # plotting
    if ax is None:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection='3d')

    # scatter centered data (easier to see PCs)
    ax.scatter(Xc[:, 0], Xc[:, 1], Xc[:, 2], s=8, alpha=0.2)

    # draw arrows for each PC from the origin (in centered coords)
    for i in range(3):
        vec = scale * np.sqrt(eigvals[i]) * V[:, i]   # length ∝ sqrt(λ_i)
        ax.quiver(0.0, 0.0, 0.0,
                  vec[0], vec[1], vec[2],
                  arrow_length_ratio=0.1, linewidth=2, color='black', alpha=1.0)
        if annotate:
            tip = 1.05 * vec
            ax.text(tip[0], tip[1], tip[2],
                    f"PC{i+1}\nλ={eigvals[i]:.3f}",
                    ha='center', va='center')

    ax.set_xlabel("x1 (centered)")
    ax.set_ylabel("x2 (centered)")
    ax.set_zlabel("x3 (centered)")
    ax.set_title("Samples and top-3 PCs (3D)")
    _set_axes_equal_3d(ax)
    return {"pcs": V, "eigvals": eigvals, "mean": mean}




# --- PCA alignment helpers ---


def best_match_align_eigenvectors(W, V=None, eigenvectors=None):
    """
    In the context of Hebbian Network, A can be two conditions:
        - A = W x (the learned components, without any lateral connections)
        - A = (I + V)W x (the learned components, with lateral connections)
    The mathematical derivation is as follows:
        - When activation function is linear Hebbian Network, Y = W x, where x is the input data.
        - When activation function is linear Hebbian Network with lateral connections, Y = (I + V)W x, where x is the input data.
    eigenvectors is the eigenvectors of the covariance matrix of the data. The eigenvalues are not needed.
    Returns:
        perm: indices of eigenvectors assigned to each W row
        corrs: absolute correlations after optimal assignment
        C: full |cosine| matrix (m x m)
    """
    W = np.asarray(W, dtype=float)
    
    # Compute A = W or A = (I + V) @ W
    if V is None:
        A = W
    else:
        V = np.asarray(V, dtype=float)
        A = (np.eye(W.shape[0]) + V) @ W

    
    eigenvectors = np.asarray(eigenvectors, dtype=float)
    
    # Normalize rows of A (each row is a weight vector)
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    
    # Normalize columns of eigenvectors (each column is an eigenvector)
    eig_norm = eigenvectors / (np.linalg.norm(eigenvectors, axis=0, keepdims=True) + 1e-12)
    
    # Compute cosine similarity matrix: C[i, j] = |cos(θ)| between A[i, :] and eigenvectors[:, j]
    C = np.abs(A_norm @ eig_norm)  # shape (m, k) where k is number of eigenvectors
    
    # Use Hungarian algorithm to find optimal matching
    # Convert similarity to cost: cost = 1 - similarity
    m, k = C.shape
    # If we have more weight vectors than eigenvectors, pad with zeros
    # If we have more eigenvectors than weight vectors, we'll match to the first m
    if m > k:
        # Pad C with zeros (no match possible for extra rows)
        C_padded = np.zeros((m, m))
        C_padded[:, :k] = C
        C_use = C_padded
    elif k > m:
        # Only use first m eigenvectors
        C_use = C[:, :m]
    else:
        C_use = C
    
    # Hungarian algorithm: minimize cost = 1 - similarity
    row_ind, col_ind = linear_sum_assignment(1.0 - C_use)
    
    # Build permutation array
    perm = np.empty(m, dtype=int)
    perm[row_ind] = col_ind
    
    # Get the actual similarities for the matched pairs
    # Handle case where we might have matched to padded columns
    corrs = np.zeros(m)
    for i in range(m):
        if perm[i] < k:  # Only if matched to a real eigenvector
            corrs[i] = C[i, perm[i]]
    
    return perm, corrs, C




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
        
    def _center(A):
        """Center columns by subtracting mean."""
        return A - A.mean(axis=0, keepdims=True)

    if metric == "corr":
        Yz, PCz = _zscore(Y), _zscore(PCs)
        # correlation matrix between columns
        S = (Yz.T @ PCz) / (n - 1)
    elif metric == "cov":
        # Center both Y and PCs
        Yc = _center(Y)
        PCc = _center(PCs)
       # Cross-covariance matrix: S[i, j] = Cov(Y[:, i], PCs[:, j])
        cov_y_pc = (Yc.T @ PCc) / (n - 1)
        # Variance of each PC column
        var_pc = np.sum(PCc ** 2, axis=0) / (n - 1)   # shape (m,)

        # Normalize: divide each column j by Var(PCs[:, j])
        S = cov_y_pc / var_pc[np.newaxis, :] 

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
            PC, _, _ = pca_topk(X, k=model.output_size)
            ordered_PCs , best_corrs, _ = best_match_alignment(W, PC)
            hist_best_corr.append(best_corrs)
            print(f"step {step:4d} | order = {ordered_PCs} | best|corr|={np.round(best_corrs,3)}")

    hist_best_corr = np.array(hist_best_corr)
    display_dict = {
                f"neuron {i+1}": hist_best_corr[:, i] for i in ordered_PCs
            }

    visualize_alignment(list(display_dict.values()), list(display_dict.keys()), model.output_size, "batch")


def Y_aligned_training_testing(model, X, batch_size=1, alignment = "corr"):
    number_of_samples = X.shape[0]
    for step in range(0, number_of_samples, batch_size):
        X_train = X[step:step+batch_size]
        model.step(X_train)
    # post-training, re-compute outputs for all samples using the learned weights
    Y_output = model.transform(X)
    # get true PC scores
    PC_scores, _, _ = pca_topk(X, k=model.output_size)
    order, similarity, _, _ = best_match_align_timeseries(Y_output, PC_scores, metric=alignment)
    # print(f"order = {order} | best|corr|= {np.round(similarity,3)}")
    return order, similarity



# ---------- Online training testing ------------------
def Y_aligned_training_testing_online(model, X, batch_size=1, alignment = "eigenvectors", graph=True):
    """
    alignment can be "eigenvectors" or "timeseries"
    """
    number_of_samples = X.shape[0]
    similarity_list = []
    # get true PC scores
    eigenvectors, _, PC_scores = pca_topk(X, k=model.output_size)
    # print(f'W: {model.W}; V: {model.V}')
    for step in range(0, number_of_samples, batch_size):
        X_train = X[step:step+batch_size]
        model.step(X_train)
        # use the X_mask to incrementally reveal the input matrix, and update the model
        Y_output = model.transform(X)
        try:
            V = model.V
        except AttributeError:
            V = None
        if alignment == "eigenvectors":
            order, similarity, _ = best_match_align_eigenvectors(model.W, V, eigenvectors[:,:model.output_size])
        elif alignment == "timeseries":
            order, similarity, _, _ = best_match_align_timeseries(Y_output, PC_scores[:,:model.output_size], metric=alignment)
        else:
            raise ValueError(f"alignment must be 'eigenvectors' or 'timeseries', got {alignment}")
        if step % 1000 == 0:
            print(f"step {step:4d} | order = {order} | best|corr|= {np.round(similarity,3)}")
        # based on the order, sort the similarity to match PC order
        combined_lists = zip(order, similarity)
        # Sort the pairs based on PC_orders
        sorted_combined_lists = sorted(combined_lists)
        # Unzip the sorted pairs
        _, ordered_similarity = zip(*sorted_combined_lists)
        similarity_list.append(ordered_similarity)
    similarity_list = np.array(similarity_list)
    if graph:
        visualize_alignment([similarity_list[:, i] for i in range(model.output_size)],
                                    [f"Neuron {i+1}" for i in range(model.output_size)],
                                    model.input_size, "trials")
    return order, similarity


# ------ Data generation helpers ------

def eig_sorted(Sigma: np.ndarray):
    w, V = np.linalg.eigh(Sigma)
    idx = np.argsort(w)[::-1]
    return w[idx], V[:, idx]

def ellipse_xy_from_cov(Sigma: np.ndarray, n_std: float = 2.0, num: int = 400):
    w, V = eig_sorted(Sigma)
    radii = n_std * np.sqrt(np.maximum(w, 0))
    t = np.linspace(0, 2*np.pi, num)
    circle = np.vstack([np.cos(t), np.sin(t)])  # 2 x num
    E = V @ np.diag(radii) @ circle
    return E[0], E[1]

def _plot_eigenvectors(ax, Sigma, color, label_prefix, alpha=0.95):
    """
    Plot BOTH eigenvectors (columns of V) scaled to 1σ (sqrt eigenvalues).
    Solid line = eigvec 1 (largest λ), dashed line = eigvec 2.
    Returns two Line2D proxy artists for the legend.
    """
    w, V = eig_sorted(Sigma)
    styles = ["-", "--"]
    labels = [f"{label_prefix} eigvec1 (λ₁)", f"{label_prefix} eigvec2 (λ₂)"]
    proxies = []
    for i in range(2):
        v = V[:, i] * np.sqrt(w[i])  # length = 1σ along that axis
        ax.plot([0, v[0]], [0, v[1]], styles[i], linewidth=2.2, color=color, alpha=alpha)
        # add a faint opposite direction to hint at axis line (optional)
        ax.plot([0, -v[0]], [0, -v[1]], styles[i], linewidth=1.2, color=color, alpha=0.35)
        # proxy for legend (so we can style the legend line exactly)
        proxies.append(Line2D([0], [0], linestyle=styles[i], color=color, lw=2.2, label=labels[i]))
    return proxies


# 1. The Rotation Matrix Generator

def generate_covariance_matrix(dim, eigenvalues, method="eigenvalues"):
    # Create the diagonal matrix
    D = np.diag(eigenvalues)
    
    # Generate random rotation Q
    Q, _ = np.linalg.qr(np.random.randn(dim, dim))
    
    # Reconstruct full matrix
    _Sigma = Q @ D @ Q.T
    Sigma = 0.5 * (_Sigma + _Sigma.T)
    
    return Sigma  # Must return (dim, dim) array

def generate_samples(d, number_of_samples):
    mu = np.full(d, 0.0)
    cov = generate_covariance_matrix(d, method='random')
    samples = np.random.multivariate_normal(mu, cov, size=number_of_samples)
    samples -= samples.mean(axis=0, keepdims=True)
    return samples


def rotation_matrix_nd(d: int, axis1: int, axis2: int, phi_rad: float) -> np.ndarray:
    R = np.eye(d)
    c, s = np.cos(phi_rad), np.sin(phi_rad)
    R[axis1, axis1] = c
    R[axis1, axis2] = -s
    R[axis2, axis1] = s
    R[axis2, axis2] = c
    return R


def instantiate_samples_by_cov(Sigma, N, m, seed=0, n_samples=10001):
    rng = np.random.default_rng(seed)
    _Z = rng.multivariate_normal(np.zeros(m), Sigma, size=n_samples)
    if N > m:
        Q, _ = np.linalg.qr(rng.standard_normal((N, m)))
        A = Q[:, :m]  # (N, m) projection matrix
        Z = _Z @ A.T
    else:
        Z = _Z
    return Z
    
### A helper function to check and correct signs of the Y_output based on the order of the PCs
def sign_check(Y_output, scores, order):
    # Check and correct signs by comparing with PC scores
    # For each aligned component, check if it correlates positively or negatively with the corresponding PC
    m = Y_output.shape[1]
    signs = np.ones(m)
    for i in range(m):
        pc_idx = order[i]  # Which PC this Y_output column is matched to
        # Compute correlation between Y_output[:, i] and scores[:, pc_idx]
        corr = np.corrcoef(Y_output[:, i], scores[:, pc_idx])[0, 1]
        # If negative correlation, flip the sign
        if corr < 0:
            signs[i] = -1
    return signs


### Rotation matrix helpers
def bar_graph_with_error_bars(df_best_corr, title):
    figure = plt.figure()
    means = df_best_corr.mean(axis=0)
    stds = df_best_corr.std(axis=0)
    plt.figure(figsize=(10, 6))
    plt.bar(means.index, means.values, yerr=stds.values, capsize=5)
    plt.ylabel('Mean Absolute Correlation')
    plt.title(title)
    plt.show()

def two_phase_training_testing(model, data_before, data_after, m, model_paras_after=None, runs =10, graph = True, alignment = "eigenvectors"):
    df_best_corr_after = pd.DataFrame(data=np.full((runs, m), np.nan), 
                            columns=[f"PC_{i+1}" for i in range(m)],
                            index=[f"run_{i+1}" for i in range(runs)])
    df_best_corr_before = pd.DataFrame(data=np.full((runs, m), np.nan), 
                            columns=[f"PC_{i+1}" for i in range(m)],
                            index=[f"run_{i+1}" for i in range(runs)])

    for run in range(runs):
        print(f"run {run+1:2d}")
        print(f"X_before")
        order, similarity = Y_aligned_training_testing_online(model, X=data_before, graph=True, alignment=alignment)
        # print(f"After first phase: model.W = {model.W}, model.V: {model.V}")

        # print(f"Y_output_after_first_phase: {model.transform(data_before)}")
       # based on the order, sort the similarity to match PC order
        combined_lists = zip(order, similarity)
        # Sort the pairs based on PC_orders
        sorted_combined_lists = sorted(combined_lists)
        # Unzip the sorted pairs
        _, ordered_similarity = zip(*sorted_combined_lists)
        df_best_corr_before.iloc[run, :] = ordered_similarity 

#################################################################
        print(f"X_after")
        # update the model parameters, if provided
        if model_paras_after is not None:
            for key, value in model_paras_after.items():
                if hasattr(model, key):
                    setattr(model, key, value)

        order, similarity = Y_aligned_training_testing_online(model, X=data_after, graph=True, alignment=alignment)

        # based on the order, sort the similarity to match PC order
        combined_lists = zip(order, similarity)
        # Sort the pairs based on PC_orders
        sorted_combined_lists = sorted(combined_lists)
        # Unzip the sorted pairs
        _, ordered_similarity = zip(*sorted_combined_lists)
        df_best_corr_after.iloc[run, :] = ordered_similarity
        # print(f"run {run+1:2d} | order = {order} | best|corr|= {np.round(similarity,3)}")
        model.reset()

    if graph == True:
        bar_graph_with_error_bars(df_best_corr_before, "Mean Absolute Correlation (before)")
        bar_graph_with_error_bars(df_best_corr_after, "Mean Absolute Correlation (after)")