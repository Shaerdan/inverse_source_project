"""
Regularization Methods for Inverse Source Localization
=======================================================

This module provides various regularization techniques for the ill-posed
inverse source problem: given boundary measurements, recover source intensities.

Methods
-------
1. L2 (Tikhonov): Minimizes ||Gq - u||² + α||q||²
2. L1 (Sparsity): Minimizes ||Gq - u||² + α||q||₁
3. TV (Total Variation): Minimizes ||Gq - u||² + α·TV(q)

The L1 method is recommended for point source recovery as it promotes
sparse solutions. TV is better for piecewise constant source distributions.
"""

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import cg, spsolve
from scipy.spatial import Delaunay
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class RegularizationResult:
    """Result from regularization solve."""
    q: np.ndarray
    residual: float
    regularization_term: float
    iterations: int
    converged: bool


def solve_l2(G: np.ndarray, u: np.ndarray, alpha: float = 1e-4) -> np.ndarray:
    """
    Solve with Tikhonov (L2) regularization.
    
    Minimizes: ||Gq - u||² + α||q||²
    
    Closed-form solution: q = (G'G + αI)⁻¹ G'u
    
    Parameters
    ----------
    G : array, shape (m, n)
        Forward operator (Green's matrix)
    u : array, shape (m,)
        Measurements
    alpha : float
        Regularization parameter
        
    Returns
    -------
    q : array, shape (n,)
        Regularized solution
    """
    n = G.shape[1]
    GtG = G.T @ G
    Gtu = G.T @ u
    q = np.linalg.solve(GtG + alpha * np.eye(n), Gtu)
    return q


def solve_l1(G: np.ndarray, u: np.ndarray, alpha: float = 1e-4,
             max_iter: int = 50, tol: float = 1e-6) -> np.ndarray:
    """
    Solve with L1 (sparsity-promoting) regularization via IRLS.
    
    Minimizes: ||Gq - u||² + α||q||₁
    
    Uses Iteratively Reweighted Least Squares (IRLS).
    
    Parameters
    ----------
    G : array, shape (m, n)
        Forward operator
    u : array, shape (m,)
        Measurements
    alpha : float
        Regularization parameter
    max_iter : int
        Maximum IRLS iterations
    tol : float
        Convergence tolerance
        
    Returns
    -------
    q : array, shape (n,)
        Sparse solution
    """
    n = G.shape[1]
    q = np.zeros(n)
    eps = 1e-4  # Smoothing parameter for |q| ≈ 0
    
    GtG = G.T @ G
    Gtu = G.T @ u
    
    for _ in range(max_iter):
        # Weight matrix W = diag(1/|q_i|)
        W = np.diag(1.0 / (np.abs(q) + eps))
        
        # Solve (G'G + αW)q = G'u
        q_new = np.linalg.solve(GtG + alpha * W, Gtu)
        
        if np.linalg.norm(q_new - q) < tol:
            break
        q = q_new
    
    return q


def build_gradient_operator(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build discrete gradient operator on triangulated mesh.
    
    Uses Delaunay triangulation to find edges, then creates
    a matrix D such that (Dq)_e = q_j - q_i for edge e = (i,j).
    
    Parameters
    ----------
    points : array, shape (n, 2)
        Node positions
        
    Returns
    -------
    D : array, shape (n_edges, n)
        Gradient operator
    edges : array, shape (n_edges, 2)
        Edge indices
    """
    tri = Delaunay(points)
    edges_set = set()
    
    for simplex in tri.simplices:
        for i in range(3):
            edge = tuple(sorted([simplex[i], simplex[(i+1) % 3]]))
            edges_set.add(edge)
    
    edges = np.array(list(edges_set))
    n_edges = len(edges)
    n_points = len(points)
    
    D = np.zeros((n_edges, n_points))
    for k, (i, j) in enumerate(edges):
        D[k, i] = 1
        D[k, j] = -1
    
    return D, edges


def solve_tv_admm(G: np.ndarray, u: np.ndarray, D: np.ndarray,
                  alpha: float = 1e-4, rho: float = 1.0,
                  max_iter: int = 100, tol: float = 1e-6,
                  verbose: bool = False) -> RegularizationResult:
    """
    Solve with Total Variation regularization using ADMM.
    
    Minimizes: (1/2)||Gq - u||² + α||Dq||₁
    
    ADMM formulation:
        min_q,z  (1/2)||Gq - u||² + α||z||₁  s.t. Dq = z
    
    Parameters
    ----------
    G : array, shape (m, n)
        Forward operator
    u : array, shape (m,)
        Measurements
    D : array, shape (n_edges, n)
        Gradient operator
    alpha : float
        Regularization parameter
    rho : float
        ADMM penalty parameter
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance
    verbose : bool
        Print convergence info
        
    Returns
    -------
    result : RegularizationResult
        Solution and convergence info
    """
    n = G.shape[1]
    n_edges = D.shape[0]
    
    # Initialize
    q = np.zeros(n)
    z = np.zeros(n_edges)
    w = np.zeros(n_edges)  # Dual variable
    
    # Precompute
    GtG = G.T @ G
    DtD = D.T @ D
    Gtu = G.T @ u
    
    # Factor for q-update: (G'G + ρD'D)⁻¹
    A = GtG + rho * DtD
    A_inv = np.linalg.inv(A)
    
    for it in range(max_iter):
        # q-update: minimize (1/2)||Gq - u||² + (ρ/2)||Dq - z + w||²
        q = A_inv @ (Gtu + rho * D.T @ (z - w))
        
        # z-update: soft thresholding
        Dq = D @ q
        z_old = z.copy()
        z = np.sign(Dq + w) * np.maximum(np.abs(Dq + w) - alpha/rho, 0)
        
        # w-update: dual ascent
        w = w + Dq - z
        
        # Check convergence
        primal_res = np.linalg.norm(Dq - z)
        dual_res = rho * np.linalg.norm(D.T @ (z - z_old))
        
        if verbose and it % 20 == 0:
            energy = 0.5 * np.linalg.norm(G @ q - u)**2 + alpha * np.sum(np.abs(D @ q))
            print(f"  ADMM iter {it}: primal_res={primal_res:.2e}, dual_res={dual_res:.2e}, energy={energy:.2e}")
        
        if primal_res < tol and dual_res < tol:
            break
    
    residual = np.linalg.norm(G @ q - u)
    reg_term = np.sum(np.abs(D @ q))
    
    return RegularizationResult(
        q=q,
        residual=residual,
        regularization_term=reg_term,
        iterations=it + 1,
        converged=(primal_res < tol and dual_res < tol)
    )


def solve_tv_chambolle_pock(G: np.ndarray, u: np.ndarray, D: np.ndarray,
                             alpha: float = 1e-4, tau: float = None,
                             sigma: float = None, max_iter: int = 200,
                             tol: float = 1e-6, verbose: bool = False) -> RegularizationResult:
    """
    Solve with TV regularization using Chambolle-Pock primal-dual algorithm.
    
    Minimizes: (1/2)||Gq - u||² + α||Dq||₁
    
    Parameters
    ----------
    G : array
        Forward operator
    u : array
        Measurements
    D : array
        Gradient operator
    alpha : float
        Regularization parameter
    tau, sigma : float
        Step sizes (auto-computed if None)
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance
    verbose : bool
        Print info
        
    Returns
    -------
    result : RegularizationResult
    """
    m, n = G.shape
    n_edges = D.shape[0]
    
    # Combine operators
    K = np.vstack([G, D])  # Stacked operator
    
    # Compute step sizes based on operator norms
    if tau is None or sigma is None:
        L = np.linalg.norm(K, 2)  # Spectral norm
        tau = 1.0 / L
        sigma = 1.0 / L
    
    # Initialize
    q = np.zeros(n)
    q_bar = q.copy()
    y = np.zeros(m + n_edges)  # Dual variable (for data fidelity and TV)
    
    for it in range(max_iter):
        q_old = q.copy()
        
        # Dual update
        Kq_bar = K @ q_bar
        y_temp = y + sigma * Kq_bar
        
        # Proximal for dual:
        # y = (y1, y2) where y1 for data fidelity, y2 for TV
        y1 = y_temp[:m]
        y2 = y_temp[m:]
        
        # Prox for (1/2)||.||² is shrinkage: y1 = y1_temp / (1 + sigma)
        y[:m] = y1 / (1 + sigma)
        
        # Prox for α||.||₁* is projection onto ||.||_∞ ≤ α
        y[m:] = np.clip(y2, -alpha, alpha)
        
        # Primal update
        q = q - tau * (K.T @ y)
        
        # Over-relaxation
        q_bar = 2 * q - q_old
        
        # Check convergence
        diff = np.linalg.norm(q - q_old) / (np.linalg.norm(q_old) + 1e-10)
        
        if verbose and it % 50 == 0:
            energy = 0.5 * np.linalg.norm(G @ q - u)**2 + alpha * np.sum(np.abs(D @ q))
            print(f"  CP iter {it}: diff={diff:.2e}, energy={energy:.2e}")
        
        if diff < tol:
            break
    
    # Shift to match measurements (remove mean)
    q = q - np.mean(q)
    
    residual = np.linalg.norm(G @ q - u)
    reg_term = np.sum(np.abs(D @ q))
    
    return RegularizationResult(
        q=q,
        residual=residual,
        regularization_term=reg_term,
        iterations=it + 1,
        converged=(diff < tol)
    )


def solve_regularized(G: np.ndarray, u: np.ndarray, 
                      points: np.ndarray = None,
                      method: str = 'l1', alpha: float = 1e-4,
                      **kwargs) -> np.ndarray:
    """
    Unified interface for regularized inverse solvers.
    
    Parameters
    ----------
    G : array
        Forward operator
    u : array
        Measurements
    points : array, optional
        Node positions (needed for TV)
    method : str
        'l1', 'l2', 'tv_admm', or 'tv_cp'
    alpha : float
        Regularization parameter
    **kwargs
        Additional arguments passed to solver
        
    Returns
    -------
    q : array
        Solution
    """
    u = u - np.mean(u)  # Center measurements
    
    if method == 'l2':
        q = solve_l2(G, u, alpha)
    elif method == 'l1':
        q = solve_l1(G, u, alpha, **kwargs)
    elif method in ('tv', 'tv_admm'):
        if points is None:
            raise ValueError("points required for TV regularization")
        D, _ = build_gradient_operator(points)
        result = solve_tv_admm(G, u, D, alpha, **kwargs)
        q = result.q
    elif method in ('tv_cp', 'chambolle_pock'):
        if points is None:
            raise ValueError("points required for TV regularization")
        D, _ = build_gradient_operator(points)
        result = solve_tv_chambolle_pock(G, u, D, alpha, **kwargs)
        q = result.q
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return q - np.mean(q)
