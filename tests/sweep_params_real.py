
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hebbian.models import OneShotLeenCompletePCA
from hebbian.utils import generate_covariance_matrix, instantiate_samples_by_cov, pca_topk

def run_joint_sweep():
    # User parameters
    N = 48
    m = 3
    n_samples = 5000
    eigenvalues = np.array([12.0, 5.0, 1.0])
    Sigma = generate_covariance_matrix(dim=m, eigenvalues=eigenvalues, method="eigenvalues")
    X = instantiate_samples_by_cov(Sigma, N, m, seed=42, n_samples=n_samples)
    pcs_gt, _, _ = pca_topk(X, k=m)

    print(f"| {'eta_v':<10} | {'C':<5} | {'V_norm':<10} | {'PC1_Sim':<10} |")
    print("-" * 50)
    
    # Base eta_v is 3e-4. Let's try 1x, 5x, 10x
    eta_v_list = [3e-4, 1e-3, 3e-3, 1e-2]
    C_list = [3.0, 10.0, 30.0]
    
    for eta_v in eta_v_list:
        for c in C_list:
            model = OneShotLeenCompletePCA(
                input_size=N, output_size=m,
                eta_w=5e-5,          
                eta_v=eta_v,          
                C=c,
                ema_alpha=0.02,
                symmetrize_V=True,
                seed=42,
                clip_V_spectral=None
            )
            
            # Fast run
            for i in range(0, n_samples, 10):
                model.step(X[i:i+10])
                
            v_norm = np.linalg.norm(model.V)
            v = model.W[0]
            sim = np.abs(np.dot(v, pcs_gt[:, 0]) / (np.linalg.norm(v) * np.linalg.norm(pcs_gt[:, 0])))
            print(f"| {eta_v:<10.2e} | {c:<5.1f} | {v_norm:<10.2f} | {sim:<10.4f} |")

if __name__ == "__main__":
    run_joint_sweep()
