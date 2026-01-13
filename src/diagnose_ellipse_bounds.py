#!/usr/bin/env python3
"""
Confirm diagnosis: Does the nonlinear solver properly constrain points to inside the ellipse?

Tests:
1. Check current implementation - is there a barrier/penalty function?
2. Test forward model behavior at points outside the ellipse
3. Test objective function at out-of-domain points
4. Verify this is the root cause of 4-source failure

Usage:
    cd src/
    python diagnose_ellipse_bounds.py
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from conformal_solver import (ConformalForwardSolver, ConformalNonlinearInverseSolver,
                               EllipseMap)


def test_current_implementation():
    """
    TEST 1: Check how bounds are currently implemented in ConformalNonlinearInverseSolver
    """
    print("="*60)
    print("TEST 1: Current Implementation Check")
    print("="*60)
    
    a, b = 2.0, 1.0
    cmap = EllipseMap(a=a, b=b)
    
    inverse = ConformalNonlinearInverseSolver(cmap, n_sources=2, n_boundary=100)
    
    # Check if there's a _objective method and what it does
    print("\nChecking ConformalNonlinearInverseSolver._objective source code...")
    
    import inspect
    try:
        source = inspect.getsource(inverse._objective)
        print("\n_objective method source (first 80 lines):")
        lines = source.split('\n')[:80]
        for i, line in enumerate(lines):
            print(f"  {i+1:3d}: {line}")
        
        # Look for penalty/barrier terms
        if 'log' in source.lower() or 'barrier' in source.lower() or 'penalty' in source.lower():
            print("\n  ✅ Found log/barrier/penalty keyword in _objective")
        else:
            print("\n  ❌ No log/barrier/penalty found in _objective")
            
        if 'is_inside' in source or 'inside' in source.lower():
            print("  ✅ Found 'inside' check in _objective")
        else:
            print("  ❌ No 'inside' check found in _objective")
            
    except Exception as e:
        print(f"  Could not get source: {e}")
    
    # Check bounds used by solver
    print("\n\nChecking solver bounds setup...")
    try:
        source = inspect.getsource(inverse.solve)
        if 'bounds' in source:
            print("  Found 'bounds' in solve method")
            # Extract bounds-related lines
            for line in source.split('\n'):
                if 'bounds' in line.lower() or 'lb' in line or 'ub' in line:
                    print(f"    {line.strip()}")
    except Exception as e:
        print(f"  Could not get source: {e}")


def test_forward_model_outside_domain():
    """
    TEST 2: What does the forward model return for points outside the ellipse?
    """
    print("\n" + "="*60)
    print("TEST 2: Forward Model Behavior Outside Domain")
    print("="*60)
    
    a, b = 2.0, 1.0
    cmap = EllipseMap(a=a, b=b)
    forward = ConformalForwardSolver(cmap, n_boundary=100)
    
    # Test points at various distances from center
    test_cases = [
        # (x, y, description)
        (0.5, 0.0, "well inside"),
        (1.0, 0.0, "inside, closer to boundary"),
        (1.5, 0.0, "inside, near boundary (x/a)²=0.56"),
        (1.8, 0.0, "inside, very near boundary (x/a)²=0.81"),
        (1.95, 0.0, "just inside boundary (x/a)²=0.95"),
        (2.0, 0.0, "ON boundary (x/a)²=1.0"),
        (2.1, 0.0, "just outside boundary (x/a)²=1.10"),
        (2.5, 0.0, "outside boundary (x/a)²=1.56"),
        (3.0, 0.0, "far outside boundary (x/a)²=2.25"),
        # Corner cases (where rectangle contains but ellipse doesn't)
        (1.5, 0.8, "rectangle corner, (x/a)²+(y/b)²=1.20, OUTSIDE ellipse"),
        (1.7, 0.7, "rectangle corner, (x/a)²+(y/b)²=1.21, OUTSIDE ellipse"),
    ]
    
    print(f"\n{'Point':<20} {'(x/a)²+(y/b)²':<15} {'Status':<12} {'||u||':<12} {'u range'}")
    print("-" * 80)
    
    for x, y, desc in test_cases:
        sources = [((x, y), 1.0), ((-x, -y), -1.0)]  # Dipole
        param = (x/a)**2 + (y/b)**2
        status = "INSIDE" if param < 1 else ("BOUNDARY" if abs(param - 1) < 0.01 else "OUTSIDE")
        
        try:
            u = forward.solve(sources)
            u_norm = np.linalg.norm(u)
            u_range = f"[{u.min():.2f}, {u.max():.2f}]"
            
            # Check for NaN/Inf
            if np.any(np.isnan(u)) or np.any(np.isinf(u)):
                u_range = "NaN/Inf!"
                
        except Exception as e:
            u_norm = "ERROR"
            u_range = str(e)[:30]
        
        print(f"({x:.1f}, {y:.1f})".ljust(20) + f"{param:.4f}".ljust(15) + f"{status}".ljust(12) + f"{u_norm if isinstance(u_norm, str) else f'{u_norm:.4f}'}".ljust(12) + f"{u_range}")


def test_objective_outside_domain():
    """
    TEST 3: What does the objective function return for out-of-domain parameters?
    """
    print("\n" + "="*60)
    print("TEST 3: Objective Function at Out-of-Domain Points")
    print("="*60)
    
    a, b = 2.0, 1.0
    cmap = EllipseMap(a=a, b=b)
    
    # Generate data from valid sources
    valid_sources = [((0.8, 0.0), 1.0), ((-0.8, 0.0), -1.0)]
    forward = ConformalForwardSolver(cmap, n_boundary=100)
    u_data = forward.solve(valid_sources)
    
    inverse = ConformalNonlinearInverseSolver(cmap, n_sources=2, n_boundary=100)
    
    # True params: [x1, y1, x2, y2, q1, q2]
    true_params = np.array([0.8, 0.0, -0.8, 0.0, 1.0, -1.0])
    
    print(f"\nTrue params: {true_params}")
    print(f"Objective at truth: {inverse._objective(true_params, u_data):.6e}")
    
    # Test with source 1 moved progressively outside
    print(f"\n{'x1':<8} {'(x1/a)²':<10} {'Status':<10} {'Objective':<15} {'Notes'}")
    print("-" * 60)
    
    for x1 in [0.8, 1.0, 1.5, 1.8, 1.95, 2.0, 2.1, 2.5, 3.0]:
        params = true_params.copy()
        params[0] = x1
        param_val = (x1/a)**2
        status = "INSIDE" if param_val < 1 else "OUTSIDE"
        
        try:
            obj = inverse._objective(params, u_data)
            if np.isnan(obj) or np.isinf(obj):
                notes = "NaN/Inf!"
            elif obj > 1e6:
                notes = "HUGE (barrier?)"
            else:
                notes = ""
        except Exception as e:
            obj = float('nan')
            notes = str(e)[:20]
        
        print(f"{x1:<8.2f} {param_val:<10.4f} {status:<10} {obj:<15.4e} {notes}")
    
    # Test corner case (in rectangle but outside ellipse)
    print(f"\nCorner case tests (inside bounding box, outside ellipse):")
    print(f"{'(x1, y1)':<15} {'(x/a)²+(y/b)²':<15} {'Objective'}")
    print("-" * 50)
    
    corner_tests = [
        (1.5, 0.8),   # (1.5/2)² + (0.8/1)² = 0.5625 + 0.64 = 1.2025
        (1.7, 0.6),   # (1.7/2)² + (0.6/1)² = 0.7225 + 0.36 = 1.0825
        (1.6, 0.7),   # (1.6/2)² + (0.7/1)² = 0.64 + 0.49 = 1.13
    ]
    
    for x1, y1 in corner_tests:
        params = true_params.copy()
        params[0] = x1
        params[1] = y1
        param_val = (x1/a)**2 + (y1/b)**2
        
        obj = inverse._objective(params, u_data)
        print(f"({x1}, {y1})".ljust(15) + f"{param_val:.4f}".ljust(15) + f"{obj:.4e}")


def test_optimization_trajectory():
    """
    TEST 4: Track what points the optimizer actually evaluates
    """
    print("\n" + "="*60)
    print("TEST 4: Optimization Trajectory (what points does optimizer try?)")
    print("="*60)
    
    a, b = 2.0, 1.0
    cmap = EllipseMap(a=a, b=b)
    
    # 4-source case that fails
    sources = [
        ((1.0, 0.0), 1.0),
        ((-1.0, 0.0), -1.0),
        ((0.0, 0.5), 1.0),
        ((0.0, -0.5), -1.0),
    ]
    
    forward = ConformalForwardSolver(cmap, n_boundary=100)
    u_data = forward.solve(sources)
    
    # Create inverse solver
    inverse = ConformalNonlinearInverseSolver(cmap, n_sources=4, n_boundary=100)
    
    # Wrap objective to track evaluations
    eval_log = []
    original_objective = inverse._objective
    
    def tracking_objective(params, u_measured):
        result = original_objective(params, u_measured)
        
        # Check if any source is outside ellipse
        n_sources = 4
        outside_count = 0
        for i in range(n_sources):
            x, y = params[2*i], params[2*i + 1]
            if (x/a)**2 + (y/b)**2 > 1.0:
                outside_count += 1
        
        eval_log.append({
            'obj': result,
            'outside_count': outside_count,
            'params': params.copy()
        })
        return result
    
    inverse._objective = tracking_objective
    
    print("\nRunning L-BFGS-B with tracking...")
    try:
        sources_rec, residual = inverse.solve(u_data, method='L-BFGS-B', n_restarts=1, seed=42)
        
        # Analyze log
        n_evals = len(eval_log)
        n_outside = sum(1 for e in eval_log if e['outside_count'] > 0)
        max_outside = max(e['outside_count'] for e in eval_log)
        
        print(f"\n  Total objective evaluations: {n_evals}")
        print(f"  Evaluations with sources OUTSIDE ellipse: {n_outside} ({100*n_outside/n_evals:.1f}%)")
        print(f"  Max sources outside in single eval: {max_outside}")
        
        # Show some examples of outside evaluations
        outside_evals = [e for e in eval_log if e['outside_count'] > 0]
        if outside_evals:
            print(f"\n  Examples of outside-domain evaluations:")
            for i, e in enumerate(outside_evals[:5]):
                params = e['params']
                print(f"    Eval: obj={e['obj']:.4e}, outside={e['outside_count']}")
                for j in range(4):
                    x, y = params[2*j], params[2*j+1]
                    p = (x/a)**2 + (y/b)**2
                    status = "OUT" if p > 1 else "in"
                    print(f"      Source {j+1}: ({x:.3f}, {y:.3f}), (x/a)²+(y/b)²={p:.3f} {status}")
                print()
                
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()


def test_manual_barrier():
    """
    TEST 5: What if we manually add a log barrier? Does it fix the problem?
    """
    print("\n" + "="*60)
    print("TEST 5: Manual Log Barrier Test")
    print("="*60)
    
    a, b = 2.0, 1.0
    cmap = EllipseMap(a=a, b=b)
    
    # 4-source case that fails
    sources = [
        ((1.0, 0.0), 1.0),
        ((-1.0, 0.0), -1.0),
        ((0.0, 0.5), 1.0),
        ((0.0, -0.5), -1.0),
    ]
    
    forward = ConformalForwardSolver(cmap, n_boundary=100)
    u_data = forward.solve(sources)
    
    inverse = ConformalNonlinearInverseSolver(cmap, n_sources=4, n_boundary=100)
    
    # Create barrier-augmented objective
    original_objective = inverse._objective
    barrier_weight = 0.01  # λ in the barrier term
    
    def barrier_objective(params, u_measured):
        base_obj = original_objective(params, u_measured)
        
        # Add log barrier for ellipse constraint
        # Constraint: (x/a)² + (y/b)² < 1
        # Barrier: -λ * log(1 - (x/a)² - (y/b)²)
        barrier = 0.0
        n_sources = 4
        for i in range(n_sources):
            x, y = params[2*i], params[2*i + 1]
            margin = 1.0 - (x/a)**2 - (y/b)**2
            
            if margin <= 0:
                # Outside or on boundary - return huge value
                return 1e10
            else:
                barrier -= barrier_weight * np.log(margin)
        
        return base_obj + barrier
    
    inverse._objective = barrier_objective
    
    print(f"\nTesting with manual log barrier (λ={barrier_weight})...")
    print("Running L-BFGS-B (n_restarts=5)...")
    
    try:
        sources_rec, residual = inverse.solve(u_data, method='L-BFGS-B', n_restarts=5, seed=42)
        
        # Compute RMSE
        from scipy.spatial.distance import cdist
        from scipy.optimize import linear_sum_assignment
        
        pos_true = np.array([[s[0][0], s[0][1]] for s in sources])
        pos_rec = np.array([[s[0][0], s[0][1]] for s in sources_rec])
        
        cost = cdist(pos_true, pos_rec)
        row_ind, col_ind = linear_sum_assignment(cost)
        matched_dist = cost[row_ind, col_ind]
        rmse = np.sqrt(np.mean(matched_dist**2))
        
        print(f"\n  Residual: {residual:.6e}")
        print(f"  Position RMSE: {rmse:.6f}")
        print(f"\n  Recovered sources:")
        for i, ((x, y), q) in enumerate(sources_rec):
            p = (x/a)**2 + (y/b)**2
            status = "OUTSIDE!" if p > 1 else "inside"
            print(f"    {i+1}: ({x:.4f}, {y:.4f}), q={q:.4f}, (x/a)²+(y/b)²={p:.3f} {status}")
        
        if rmse < 0.1:
            print(f"\n  ✅ BARRIER FIX WORKS! RMSE improved from ~0.56 to {rmse:.4f}")
        else:
            print(f"\n  ❌ Barrier didn't fully fix it. May need tuning or different approach.")
            
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    test_current_implementation()
    test_forward_model_outside_domain()
    test_objective_outside_domain()
    test_optimization_trajectory()
    test_manual_barrier()
    
    print("\n" + "="*60)
    print("DIAGNOSIS COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
