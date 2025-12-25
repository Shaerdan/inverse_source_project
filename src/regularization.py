"""
Regularization Methods for Inverse Source Localization
=======================================================

Implements L1, L2, and Total Variation regularization with multiple algorithms.

Regularization Types:
- L2 (Tikhonov): ||q||₂² - Smooth solutions
- L1 (Sparsity): ||q||₁ - Sparse solutions (IRLS algorithm)
- TV (Total Variation): ||∇q||₁ - Piecewise constant solutions

TV Algorithms:
- Chambolle-Pock: Primal-dual algorithm (faster, better convergence)
- ADMM: Alternating Direction Method of Multipliers
"""

import numpy as np
from typing import Tuple, Optional, Callable
from dataclasses import dataclass


@dataclass
class RegularizationResult:
    """Container for regularization results."""
    q: np.ndarray
    residual: float
    reg_norm: float
    n_iter: int
    converged: bool
    history: Optional[list] = None


# =============================================================================
# L2 (TIKHONOV) REGULARIZATION
# =============================================================================

def solve_l2(G: np.ndarray, u: np.ndarray, alpha: float = 1e-4) -> RegularizationResult:
    """
    Solve with L2 (Tikhonov) regularization.
    
    min ||Gq - u||₂² + α||q||₂²
    
    Closed-form solution: q = (G'G + αI)⁻¹ G'u
    
    Parameters
    ----------
    G : array, shape (m, n)
        Forward operator (Green's matrix)
    u : array, shape (m,)
        Measured data
    alpha : float
        Regularization parameter
        
    Returns
    -------
    result : RegularizationResult
    """
    u_centered = u - np.mean(u)
    n = G.shape[1]
    
    GtG = G.T @ G
    Gtu = G.T @ u_centered
    
    q = np.linalg.solve(GtG + alpha * np.eye(n), Gtu)
    q = q - np.mean(q)  # Enforce compatibility
    
    residual = np.linalg.norm(G @ q - u_centered)
    reg_norm = np.linalg.norm(q)
    
    return RegularizationResult(
        q=q,
        residual=residual,
        reg_norm=reg_norm,
        n_iter=1,
        converged=True,
    )


# =============================================================================
# L1 (SPARSITY) REGULARIZATION - IRLS
# =============================================================================

def solve_l1(G: np.ndarray, u: np.ndarray, alpha: float = 1e-4,
             max_iter: int = 50, tol: float = 1e-6,
             verbose: bool = False) -> RegularizationResult:
    """
    Solve with L1 (sparsity) regularization via IRLS.
    
    min ||Gq - u||₂² + α||q||₁
    
    IRLS: Iteratively Reweighted Least Squares
    At each iteration, solve: (G'G + αW)q = G'u
    where W = diag(1/|q| + ε)
    
    Parameters
    ----------
    G : array, shape (m, n)
        Forward operator
    u : array, shape (m,)
        Measured data
    alpha : float
        Regularization parameter
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance
    verbose : bool
        Print convergence info
        
    Returns
    -------
    result : RegularizationResult
    """
    u_centered = u - np.mean(u)
    n = G.shape[1]
    
    q = np.zeros(n)
    eps = 1e-4
    history = []
    
    GtG = G.T @ G
    Gtu = G.T @ u_centered
    
    for k in range(max_iter):
        # Weight matrix
        W = np.diag(1.0 / (np.abs(q) + eps))
        
        # Solve weighted system
        q_new = np.linalg.solve(GtG + alpha * W, Gtu)
        q_new = q_new - np.mean(q_new)
        
        # Check convergence
        change = np.linalg.norm(q_new - q) / (np.linalg.norm(q) + 1e-14)
        history.append(change)
        
        if verbose and k % 10 == 0:
            print(f"  IRLS iter {k}: change = {change:.2e}")
        
        if change < tol:
            if verbose:
                print(f"  IRLS converged at iteration {k}")
            return RegularizationResult(
                q=q_new,
                residual=np.linalg.norm(G @ q_new - u_centered),
                reg_norm=np.sum(np.abs(q_new)),
                n_iter=k + 1,
                converged=True,
                history=history,
            )
        
        q = q_new
    
    if verbose:
        print(f"  IRLS did not converge after {max_iter} iterations")
    
    return RegularizationResult(
        q=q,
        residual=np.linalg.norm(G @ q - u_centered),
        reg_norm=np.sum(np.abs(q)),
        n_iter=max_iter,
        converged=False,
        history=history,
    )


# =============================================================================
# TOTAL VARIATION REGULARIZATION - CHAMBOLLE-POCK
# =============================================================================

def build_gradient_operator(n_radial: int, n_angular: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build discrete gradient operators for polar grid.
    
    Parameters
    ----------
    n_radial : int
        Number of radial divisions
    n_angular : int
        Number of angular divisions
        
    Returns
    -------
    Dr : array
        Radial gradient operator
    Dtheta : array
        Angular gradient operator
    """
    n = n_radial * n_angular
    
    # Radial gradient (forward difference)
    Dr = np.zeros((n, n))
    for i in range(n_radial - 1):
        for j in range(n_angular):
            idx = i * n_angular + j
            idx_next = (i + 1) * n_angular + j
            Dr[idx, idx] = -1
            Dr[idx, idx_next] = 1
    
    # Angular gradient (circular boundary)
    Dtheta = np.zeros((n, n))
    for i in range(n_radial):
        for j in range(n_angular):
            idx = i * n_angular + j
            idx_next = i * n_angular + ((j + 1) % n_angular)
            Dtheta[idx, idx] = -1
            Dtheta[idx, idx_next] = 1
    
    return Dr, Dtheta


def solve_tv_chambolle_pock(G: np.ndarray, u: np.ndarray, 
                            D: np.ndarray, alpha: float = 1e-4,
                            tau: float = 0.1, sigma: float = 0.1, theta: float = 1.0,
                            max_iter: int = 500, tol: float = 1e-5,
                            verbose: bool = False,
                            callback: Optional[Callable] = None) -> RegularizationResult:
    """
    Solve with TV regularization using Chambolle-Pock algorithm.
    
    min ||Gq - u||₂² + α||Dq||₁
    
    where D is a gradient operator (finite difference matrix).
    
    Chambolle-Pock primal-dual algorithm:
    - Primal: q^{k+1} = prox_{τf}(q^k - τD'p^k)
    - Dual: p^{k+1} = prox_{σg*}(p^k + σD(2q^{k+1} - q^k))
    
    References:
    - Chambolle, A., & Pock, T. (2011). A first-order primal-dual algorithm.
    
    Parameters
    ----------
    G : array, shape (m, n)
        Forward operator
    u : array, shape (m,)
        Measured data
    D : array, shape (k, n)
        Gradient operator
    alpha : float
        TV regularization parameter
    tau, sigma : float
        Step sizes (should satisfy τσ||D||² < 1)
    theta : float
        Extrapolation parameter (typically 1)
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance
    verbose : bool
        Print progress
    callback : callable, optional
        Function called each iteration: callback(k, q, p, primal_energy)
        
    Returns
    -------
    result : RegularizationResult
    """
    u_centered = u - np.mean(u)
    m, n = G.shape
    k_dim = D.shape[0]
    
    # Initialize
    q = np.zeros(n)
    q_bar = q.copy()
    p = np.zeros(k_dim)
    
    # Precompute
    GtG = G.T @ G
    Gtu = G.T @ u_centered
    DtD = D.T @ D
    
    history = []
    
    for k in range(max_iter):
        # Store old q
        q_old = q.copy()
        
        # Primal update: q = (I + τG'G)⁻¹(q_old - τD'p + τG'u)
        # Simplified proximal step for quadratic data term
        rhs = q_old - tau * (D.T @ p) + tau * Gtu
        q = np.linalg.solve(np.eye(n) + tau * GtG, rhs)
        q = q - np.mean(q)  # Enforce compatibility
        
        # Extrapolation
        q_bar = q + theta * (q - q_old)
        
        # Dual update: p = prox_{σα||·||₁}(p + σDq_bar)
        p_tilde = p + sigma * (D @ q_bar)
        # Soft thresholding
        p = np.sign(p_tilde) * np.maximum(np.abs(p_tilde) - sigma * alpha, 0)
        # Project to ||p||∞ ≤ α (dual of TV)
        p = np.clip(p, -alpha, alpha)
        
        # Compute primal energy (objective value)
        residual = np.linalg.norm(G @ q - u_centered)
        tv_norm = np.sum(np.abs(D @ q))
        energy = 0.5 * residual**2 + alpha * tv_norm
        history.append(energy)
        
        if callback is not None:
            callback(k, q, p, energy)
        
        # Check convergence
        if k > 0:
            rel_change = abs(history[-1] - history[-2]) / (abs(history[-2]) + 1e-14)
            if rel_change < tol:
                if verbose:
                    print(f"  Chambolle-Pock converged at iteration {k}")
                return RegularizationResult(
                    q=q,
                    residual=residual,
                    reg_norm=tv_norm,
                    n_iter=k + 1,
                    converged=True,
                    history=history,
                )
        
        if verbose and k % 50 == 0:
            print(f"  CP iter {k}: energy = {energy:.6e}, residual = {residual:.6e}")
    
    if verbose:
        print(f"  Chambolle-Pock did not converge after {max_iter} iterations")
    
    return RegularizationResult(
        q=q,
        residual=np.linalg.norm(G @ q - u_centered),
        reg_norm=np.sum(np.abs(D @ q)),
        n_iter=max_iter,
        converged=False,
        history=history,
    )


# =============================================================================
# TOTAL VARIATION REGULARIZATION - ADMM
# =============================================================================

def solve_tv_admm(G: np.ndarray, u: np.ndarray,
                  D: np.ndarray, alpha: float = 1e-4, rho: float = 1.0,
                  max_iter: int = 500, tol: float = 1e-5,
                  verbose: bool = False,
                  callback: Optional[Callable] = None) -> RegularizationResult:
    """
    Solve with TV regularization using ADMM.
    
    min ||Gq - u||₂² + α||Dq||₁
    
    ADMM formulation (with z = Dq):
    - q-update: (G'G + ρD'D)q = G'u + ρD'(z - w)
    - z-update: z = shrink(Dq + w, α/ρ)
    - w-update: w = w + Dq - z
    
    Parameters
    ----------
    G : array, shape (m, n)
        Forward operator
    u : array, shape (m,)
        Measured data
    D : array, shape (k, n)
        Gradient operator
    alpha : float
        TV regularization parameter
    rho : float
        ADMM penalty parameter
    max_iter : int
    tol : float
    verbose : bool
    callback : callable, optional
        
    Returns
    -------
    result : RegularizationResult
    """
    u_centered = u - np.mean(u)
    m, n = G.shape
    k_dim = D.shape[0]
    
    # Initialize
    q = np.zeros(n)
    z = np.zeros(k_dim)
    w = np.zeros(k_dim)  # Scaled dual variable
    
    # Precompute factorization
    GtG = G.T @ G
    DtD = D.T @ D
    Gtu = G.T @ u_centered
    A = GtG + rho * DtD
    
    history = []
    
    for k in range(max_iter):
        # q-update
        rhs = Gtu + rho * D.T @ (z - w)
        q = np.linalg.solve(A, rhs)
        q = q - np.mean(q)  # Compatibility
        
        # z-update (soft thresholding)
        Dq_plus_w = D @ q + w
        z = np.sign(Dq_plus_w) * np.maximum(np.abs(Dq_plus_w) - alpha / rho, 0)
        
        # w-update
        w = w + D @ q - z
        
        # Compute objective
        residual = np.linalg.norm(G @ q - u_centered)
        tv_norm = np.sum(np.abs(D @ q))
        energy = 0.5 * residual**2 + alpha * tv_norm
        history.append(energy)
        
        if callback is not None:
            callback(k, q, z, w, energy)
        
        # Check convergence
        if k > 0:
            rel_change = abs(history[-1] - history[-2]) / (abs(history[-2]) + 1e-14)
            if rel_change < tol:
                if verbose:
                    print(f"  ADMM converged at iteration {k}")
                return RegularizationResult(
                    q=q,
                    residual=residual,
                    reg_norm=tv_norm,
                    n_iter=k + 1,
                    converged=True,
                    history=history,
                )
        
        if verbose and k % 50 == 0:
            print(f"  ADMM iter {k}: energy = {energy:.6e}")
    
    if verbose:
        print(f"  ADMM did not converge after {max_iter} iterations")
    
    return RegularizationResult(
        q=q,
        residual=np.linalg.norm(G @ q - u_centered),
        reg_norm=np.sum(np.abs(D @ q)),
        n_iter=max_iter,
        converged=False,
        history=history,
    )


# =============================================================================
# UNIFIED INTERFACE
# =============================================================================

def solve_regularized(G: np.ndarray, u: np.ndarray, 
                      method: str = 'l1', alpha: float = 1e-4,
                      D: Optional[np.ndarray] = None,
                      **kwargs) -> RegularizationResult:
    """
    Unified interface for regularized inverse problems.
    
    Parameters
    ----------
    G : array
        Forward operator
    u : array
        Measured data
    method : str
        'l1', 'l2', 'tv_cp' (Chambolle-Pock), 'tv_admm'
    alpha : float
        Regularization parameter
    D : array, optional
        Gradient operator (required for TV methods)
    **kwargs : 
        Additional parameters passed to specific solver
        
    Returns
    -------
    result : RegularizationResult
    """
    method = method.lower()
    
    if method == 'l2':
        return solve_l2(G, u, alpha)
    
    elif method == 'l1':
        return solve_l1(G, u, alpha, **kwargs)
    
    elif method in ('tv', 'tv_cp', 'chambolle_pock'):
        if D is None:
            raise ValueError("Gradient operator D required for TV regularization")
        return solve_tv_chambolle_pock(G, u, D, alpha, **kwargs)
    
    elif method == 'tv_admm':
        if D is None:
            raise ValueError("Gradient operator D required for TV regularization")
        return solve_tv_admm(G, u, D, alpha, **kwargs)
    
    else:
        raise ValueError(f"Unknown method: {method}. "
                        f"Available: l1, l2, tv_cp, tv_admm")


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Create simple test problem
    np.random.seed(42)
    n = 100
    m = 50
    
    G = np.random.randn(m, n) / np.sqrt(m)
    q_true = np.zeros(n)
    q_true[20] = 1.0
    q_true[60] = -1.0  # Sparse solution
    
    u = G @ q_true + 0.01 * np.random.randn(m)
    
    # Build gradient operator
    D = np.eye(n) - np.roll(np.eye(n), 1, axis=1)
    
    print("Testing regularization methods...")
    
    # L2
    result_l2 = solve_l2(G, u, alpha=1e-3)
    print(f"L2: residual = {result_l2.residual:.4f}, ||q|| = {result_l2.reg_norm:.4f}")
    
    # L1
    result_l1 = solve_l1(G, u, alpha=1e-3, verbose=False)
    print(f"L1: residual = {result_l1.residual:.4f}, ||q||₁ = {result_l1.reg_norm:.4f}")
    
    # TV (Chambolle-Pock)
    result_tv_cp = solve_tv_chambolle_pock(G, u, D, alpha=1e-3, verbose=False)
    print(f"TV-CP: residual = {result_tv_cp.residual:.4f}, TV = {result_tv_cp.reg_norm:.4f}")
    
    # TV (ADMM)
    result_tv_admm = solve_tv_admm(G, u, D, alpha=1e-3, verbose=False)
    print(f"TV-ADMM: residual = {result_tv_admm.residual:.4f}, TV = {result_tv_admm.reg_norm:.4f}")
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].stem(q_true, label='True')
    axes[0, 0].stem(result_l2.q, linefmt='r-', markerfmt='ro', label='L2', basefmt=' ')
    axes[0, 0].set_title('L2 Regularization')
    axes[0, 0].legend()
    
    axes[0, 1].stem(q_true, label='True')
    axes[0, 1].stem(result_l1.q, linefmt='r-', markerfmt='ro', label='L1', basefmt=' ')
    axes[0, 1].set_title('L1 Regularization')
    axes[0, 1].legend()
    
    axes[1, 0].stem(q_true, label='True')
    axes[1, 0].stem(result_tv_cp.q, linefmt='r-', markerfmt='ro', label='TV-CP', basefmt=' ')
    axes[1, 0].set_title('TV (Chambolle-Pock)')
    axes[1, 0].legend()
    
    axes[1, 1].stem(q_true, label='True')
    axes[1, 1].stem(result_tv_admm.q, linefmt='r-', markerfmt='ro', label='TV-ADMM', basefmt=' ')
    axes[1, 1].set_title('TV (ADMM)')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('regularization_comparison.png', dpi=150)
    print("\nSaved: regularization_comparison.png")
    plt.show()
