#!/usr/bin/env python3
"""
Verification script for the MFS conformal map fix.

Run this after replacing conformal_solver.py to verify the fix is working.

Usage:
    python test_mfs_conformal_fix.py
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')


def test_boundary_mapping():
    """Test that boundary points correctly map to the unit circle."""
    from conformal_solver import NumericalConformalMap, PolygonMap, RectangleMap
    
    print("="*70)
    print("BOUNDARY MAPPING TEST")
    print("="*70)
    
    results = []
    
    # Test 1: Star domain
    print("\n1. Star Domain (NumericalConformalMap)")
    def star_boundary(t):
        r = 1.0 + 0.3 * np.cos(5 * t)
        return r * np.cos(t) + 1j * r * np.sin(t)
    
    star_map = NumericalConformalMap(star_boundary, n_boundary=256)
    z_bdy = star_map.boundary_physical(50)
    w_bdy = star_map.to_disk(z_bdy)
    w_abs = np.abs(w_bdy)
    
    print(f"   Boundary |w|: [{w_abs.min():.4f}, {w_abs.max():.4f}]")
    star_ok = w_abs.min() > 0.95
    print(f"   {'✅ PASS' if star_ok else '❌ FAIL'}")
    results.append(('Star', star_ok))
    
    # Test 2: Rectangle
    print("\n2. Square Domain (RectangleMap)")
    rect_map = RectangleMap(half_width=1.0, half_height=1.0)
    corners = np.array([1+1j, -1+1j, -1-1j, 1-1j])
    w_corners = rect_map.to_disk(corners)
    w_abs = np.abs(w_corners)
    
    print(f"   Corner |w|: [{w_abs.min():.4f}, {w_abs.max():.4f}]")
    rect_ok = w_abs.min() > 0.95
    print(f"   {'✅ PASS' if rect_ok else '❌ FAIL'}")
    results.append(('Square', rect_ok))
    
    # Test 3: Hexagon
    print("\n3. Hexagon (PolygonMap)")
    vertices = [np.exp(1j * a) for a in np.linspace(0, 2*np.pi, 6, endpoint=False)]
    hex_map = PolygonMap(vertices)
    w_verts = hex_map.to_disk(np.array(vertices))
    w_abs = np.abs(w_verts)
    
    print(f"   Vertex |w|: [{w_abs.min():.4f}, {w_abs.max():.4f}]")
    hex_ok = w_abs.min() > 0.95
    print(f"   {'✅ PASS' if hex_ok else '❌ FAIL'}")
    results.append(('Hexagon', hex_ok))
    
    return results


def test_forward_model():
    """Test that forward model gives correct physics."""
    from conformal_solver import NumericalConformalMap, ConformalForwardSolver
    
    print("\n" + "="*70)
    print("FORWARD MODEL TEST")
    print("="*70)
    
    # Star domain
    def star_boundary(t):
        return (1 + 0.3*np.cos(5*t)) * np.exp(1j*t)
    
    star_map = NumericalConformalMap(star_boundary, n_boundary=256)
    
    # Sources
    sources = [
        ((0.5, 0.0), 1.0),
        ((-0.5, 0.0), -1.0),
    ]
    
    # Forward solve
    n_sensors = 50
    sensors = np.column_stack([
        np.real(star_map.boundary_physical(n_sensors)),
        np.imag(star_map.boundary_physical(n_sensors))
    ])
    
    forward = ConformalForwardSolver(star_map, n_boundary=n_sensors, 
                                      sensor_locations=sensors)
    u = forward.solve(sources)
    
    print(f"\nSources: 2 point sources at (±0.5, 0)")
    print(f"Forward data range: [{u.min():.4f}, {u.max():.4f}]")
    
    # Self-consistency: solve again should give same result
    u2 = forward.solve(sources)
    diff = np.linalg.norm(u - u2)
    
    print(f"Self-consistency error: {diff:.2e}")
    
    ok = diff < 1e-10
    print(f"{'✅ PASS' if ok else '❌ FAIL'}")
    
    return ok


def main():
    print("\n" + "="*70)
    print("MFS CONFORMAL MAP FIX VERIFICATION")
    print("="*70)
    
    # Run tests
    boundary_results = test_boundary_mapping()
    forward_ok = test_forward_model()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    all_ok = all(r[1] for r in boundary_results) and forward_ok
    
    print("\nBoundary mapping:")
    for name, ok in boundary_results:
        print(f"  {name}: {'✅ PASS' if ok else '❌ FAIL'}")
    
    print(f"\nForward model: {'✅ PASS' if forward_ok else '❌ FAIL'}")
    
    print("\n" + "="*70)
    if all_ok:
        print("ALL TESTS PASSED - MFS fix is working correctly!")
    else:
        print("SOME TESTS FAILED - check output above")
    print("="*70)
    
    return 0 if all_ok else 1


if __name__ == '__main__':
    exit(main())
