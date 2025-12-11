
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hebbian.models import OneShotLeenCompletePCA
from hebbian.utils import generate_covariance_matrix, instantiate_samples_by_cov, pca_topk

def run_sweep():
    # User parameters
    N = 48
    m = 3
    n_samples = 5000
    eigenvalues = np.array([12.0, 5.0, 1.0])
    Sigma = generate_covariance_matrix(dim=m, eigenvalues=eigenvalues, method="eigenvalues")
    X = instantiate_samples_by_cov(Sigma, N, m, seed=0, n_samples=n_samples)
    pcs_gt, _, _ = pca_topk(X, k=m)

    print("--- Sweeping C with REAL library code ---")
    
    for C_value in [3.0, 10.0, 20.0, 30.0, 50.0]:
        model = OneShotLeenCompletePCA(
            input_size=N, output_size=m,
            eta_w=5e-5,          
            eta_v=3e-4,          
            C=C_value,
            ema_alpha=0.02,
            symmetrize_V=True,
            seed=42,
            clip_V_spectral=None
        )
        
        history_sim = []
        for i in range(0, n_samples, 10):
            model.step(X[i:i+10])
            
        v_norm = np.linalg.norm(model.V)
        v = model.W[0]
        sim = np.abs(np.dot(v, pcs_gt[:, 0]) / (np.linalg.norm(v) * np.linalg.norm(pcs_gt[:, 0])))
        print(f"C={C_value}: Final V_norm={v_norm:.2f}, PC1_Sim={sim:.4f}")

if __name__ == "__main__":
    run_sweep()
