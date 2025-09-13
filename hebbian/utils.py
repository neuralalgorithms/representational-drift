# -- coding: utf-8 --
"""
Functions used to assist training and testing Hebbian models.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

    return pcs, eigvals

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

def testing_alignment(model, X, epochs, pcs, shuffle=False, method="epochs"):
    """
    Record cosine alignment(s) while training, either:
      - per epoch  (method='epochs'): one snapshot after each epoch
      - per trials (method='trials'): snapshot after each step
    """
    X = np.asarray(X, float)
    N = len(X)
    align_hist = []

    def align_from_W(W):
        # single-output
        if W.ndim == 1 or (W.ndim == 2 and W.shape[0] == 1):
            w = W if W.ndim == 1 else W[0, :]
            return cosine_alignment(w, pcs[0])
        else:
            m = min(W.shape[0], len(pcs))
            return [cosine_alignment(W[i, :], pcs[i]) for i in range(m)]

    for _ in range(int(epochs)):
        if method == "epochs":
            model.train(X, epochs=1, shuffle=shuffle)  # one epoch
            align_hist.append(align_from_W(model.W))
        elif method == "trials":
            idx = np.arange(N)
            if shuffle:
                model.rng.shuffle(idx)
            for i in idx:
                model.step(X[i])
                align_hist.append(align_from_W(model.W))
        else:
            raise ValueError("method must be 'epochs' or 'trials'.")
    return align_hist


# the main function to train and test the models
def training_testing(samples, model1, model2, output_size=None, learning_rate=0.1, epochs=1, method = "epochs", pc_method = "ordered", show_num = None):
    global weights_model1_K, weights_model2_K
    samples_dim = samples.shape[1]
    # get the PCs 
    pcs, _ = pca_topk(samples, samples_dim, center=True)

    if pc_method == "first":
        pc_set = [pcs[:,0] for i in range(samples_dim)]
    elif pc_method == "ordered":
        pc_set = [pcs[:,i] for i in range(samples_dim)]
    else:
        raise ValueError("Has to be 'first' or 'ordered'")

    # oja1 = OjaNetwork(input_size=samples_dim, output_size=1, learning_rate=learning_rate)
    model1_K = model1(input_size=samples_dim, output_size=output_size, learning_rate=learning_rate)
    model2_K = model2(input_size=samples_dim, output_size=output_size, learning_rate=learning_rate)

    # hist_oja1 = testing_alignment(oja1, samples, epochs=epochs, pcs = [pc_set[0]], method=method)

    if output_size > samples_dim:
        hist_model1_K = testing_alignment(model1_K, samples, epochs=epochs, pcs = pc_set[:samples_dim], method=method)
        hist_model2_K = testing_alignment(model2_K, samples, epochs=epochs, pcs = pc_set[:samples_dim], method=method)
    else:
        hist_model1_K = testing_alignment(model1_K, samples, epochs=epochs, pcs = pc_set, method=method)
        hist_model2_K = testing_alignment(model2_K, samples, epochs=epochs, pcs = pc_set, method=method)

    # weights_oja1 = oja1.weights.copy()
    weights_model1_K = model1_K.weights.copy()
    weights_model2_K = model2_K.weights.copy()

    ### visualize the alignment
    display_dict_oja ={}
    display_dict_sanger ={}
    # display_dict_oja1 = {}
    # display_dict_oja1["Oja-1"] = np.array(hist_oja1)

    if show_num == None:
        show_num = output_size
    for i in range(show_num):
        display_dict_oja[f"Oja-multi (neuron {i+1})"] = np.array(hist_model1_K)[:,i]
        display_dict_sanger[f"Sanger (neuron {i+1})"] = np.array(hist_model2_K)[:,i]

    visualize_alignment(list(display_dict_oja.values()), list(display_dict_oja.keys()), input_size=samples_dim, xlabel=method)
    visualize_alignment(list(display_dict_sanger.values()), list(display_dict_sanger.keys()), input_size=samples_dim, xlabel=method)


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
# Data simulation functions
#--------------------------------

def gauss_distrib(theta, mu1, std1, mu2, std2, number_of_samples, number_of_neurons=2):
    samples = np.zeros((number_of_samples, number_of_neurons));
    for i in range(number_of_samples):
        u1 = np.random.random();
        u2 = np.random.random();
        T1 = np.sqrt(-2 * np.log(u1)) * np.cos(2*np.pi*u2);
        T2 = np.sqrt(-2 * np.log(u1)) * np.sin(2*np.pi*u2);
        
        samples[i,0] = mu1 +  (std1*T1 * np.cos(theta) - std2*T2 * np.sin(theta));
        samples[i,1] = mu2 +  (std1*T1 * np.sin(theta) + std2*T2 * np.cos(theta));	
    samples[:,0] -= np.mean(samples[:,0])
    samples[:,1] -=np.mean(samples[:,1])
    return samples

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


def main():
    """Run example usage of the tools."""
    # Example usage code here
    pass

if __name__ == "__main__":
    main()