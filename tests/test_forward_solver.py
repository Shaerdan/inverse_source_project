"""
Basic tests for the forward solver.
"""

import numpy as np
import pytest


def test_import():
    """Test that the package can be imported."""
    from src import forward_solver
    from src import mesh_utils
    assert forward_solver is not None
    assert mesh_utils is not None


def test_sources_balance():
    """Test that sources sum to zero."""
    sources = [
        ((0.0, 0.0), 1.0),
        ((0.5, 0.5), -1.0),
    ]
    total = sum(q for _, q in sources)
    assert abs(total) < 1e-10, "Sources must be balanced"


# Add more tests when FEniCSx is available
@pytest.mark.skipif(True, reason="Requires FEniCSx environment")
def test_solve_poisson():
    """Test the Poisson solver."""
    from src.forward_solver import create_disk_mesh, solve_poisson_zero_neumann
    
    mesh, _, _ = create_disk_mesh(radius=1.0, resolution=0.1)
    sources = [
        ((0.3, 0.3), 1.0),
        ((-0.3, -0.3), -1.0),
    ]
    
    u = solve_poisson_zero_neumann(mesh, sources)
    
    # Solution should exist
    assert u is not None
    assert len(u.x.array) > 0
    
    # Mean should be approximately zero
    assert abs(u.x.array.mean()) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
