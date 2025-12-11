
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hebbian.models import OneShotLeenCompletePCA
from hebbian.utils import generate_covariance_matrix, instantiate_samples_by_cov, pca_topk, best_match_align_eigenvectors

def test_convergence(sign_flip=False, C_value=3.0):
    # User parameters
    N = 48
    m = 3
    n_samples = 5000
    params = np.array([0.1, 0.7, 0.4])
    
    # Data generation
    eigenvalues = np.array([12.0, 5.0, 1.0])
    Sigma = generate_covariance_matrix(dim=m, eigenvalues=eigenvalues, method="eigenvalues")
    X = instantiate_samples_by_cov(Sigma, N, m, seed=0, n_samples=n_samples)
    
    # GT PCA
    pcs_gt, _, _ = pca_topk(X, k=m)

    # Model parameters
    leen_paras = dict(
        input_size=N, output_size=m,
        eta_w=5e-5,          
        eta_v=3e-4,          
        C=C_value, # <--- Tuning this
        ema_alpha=0.02,
        symmetrize_V=True,
        seed=42,
        clip_V_spectral=None
    )
    
    model = OneShotLeenCompletePCA(**leen_paras)
    
    # Monkeypatch for testing
    def patched_step(self, X):
        X = np.asarray(X, dtype=float)
        B = X.shape[0]
        Y = self._forward(X)
        y_cov = (Y.T @ Y) / B
        y_var = np.diag(y_cov).copy()
        self.lam = (1.0 - self.ema_alpha) * self.lam + self.ema_alpha * y_var
        
        lam_sum = self.lam[:, None] + self.lam[None, :]
        
        if sign_flip:
             # STABLE (-V)
            dV = self.eta_v * (- lam_sum * self.V - self.C * y_cov)
        else:
             # UNSTABLE (+V)
            dV = self.eta_v * (lam_sum * self.V - self.C * y_cov)
            
        np.fill_diagonal(dV, 0.0)
        self.V += dV
        
        if self.symmetrize_V:
            self._symmetrize_V()
        # No spectral clipping for this raw physics test
        
        hebb_term = (Y.T @ X) / B
        oja_term = y_var[:, None] * self.W
        dW = self.eta_w * (hebb_term - oja_term)
        self.W += dW

    import types
    model.step = types.MethodType(patched_step, model)
    
    # Run loop
    batch_size = 10
    history_V_norm = []
    vals = []
    
    for i in range(0, n_samples, batch_size):
        batch = X[i:i+batch_size]
        model.step(batch)
        history_V_norm.append(np.linalg.norm(model.V))
        
        if i % 100 == 0:
             v = model.W[0]
             sim = np.abs(np.dot(v, pcs_gt[:, 0]) / (np.linalg.norm(v) * np.linalg.norm(pcs_gt[:, 0])))
             vals.append(sim)

    return history_V_norm, vals

print("Running Unstable (+V) Baseline (C=3.0)...")
v_unstable, acc_unstable = test_convergence(sign_flip=False, C_value=3.0)

print("\n--- Tuning Stable (-V) ---")
for c in [3.0, 6.0, 10.0, 15.0]:
    v, acc = test_convergence(sign_flip=True, C_value=c)
    print(f"Stable (-V), C={c}: Final V_norm={v[-1]:.2f}, PC1_Sim={acc[-1]:.4f}")

