"""
Forward Solver for Poisson Equation with Point Sources
======================================================
Solves: -Δu = f  in Ω (unit disk)
        ∂u/∂n = 0  on ∂Ω (zero Neumann BC)

Uses FEniCSx (dolfinx 0.10+) with scipy solver
"""

import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

# FEniCSx imports
import dolfinx
from dolfinx import default_scalar_type
from dolfinx.fem import functionspace, Function, form, assemble_scalar, Constant
from dolfinx.fem.petsc import assemble_matrix
from dolfinx.io.gmsh import model_to_mesh
import ufl
from ufl import TrialFunction, TestFunction, grad, inner, dx

import gmsh


def create_disk_mesh(radius: float = 1.0, resolution: float = 0.05):
    """Create a disk mesh using Gmsh."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("disk")
    
    disk = gmsh.model.occ.addDisk(0, 0, 0, radius, radius)
    gmsh.model.occ.synchronize()
    
    # Physical groups required by dolfinx 0.10+
    boundary = gmsh.model.getBoundary([(2, disk)], oriented=False)
    boundary_tags = [b[1] for b in boundary]
    gmsh.model.addPhysicalGroup(1, boundary_tags, 1, name="boundary")
    gmsh.model.addPhysicalGroup(2, [disk], 1, name="domain")
    
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", resolution)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", resolution)
    gmsh.model.mesh.generate(2)
    
    mesh_data = model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
    gmsh.finalize()
    
    return mesh_data[0], mesh_data[1], mesh_data[2]


def create_source_function(V, sources, sigma=0.05):
    """
    Create a source function as sum of Gaussians (regularized deltas).
    """
    f = Function(V)
    x = V.tabulate_dof_coordinates()
    
    f_values = np.zeros(len(x))
    for (x0, y0), intensity in sources:
        r2 = (x[:, 0] - x0)**2 + (x[:, 1] - y0)**2
        f_values += intensity * np.exp(-r2 / (2 * sigma**2)) / (2 * np.pi * sigma**2)
    
    f.x.array[:] = f_values
    return f


def petsc_to_scipy(A):
    """Convert PETSc matrix to scipy sparse matrix."""
    ai, aj, av = A.getValuesCSR()
    return csr_matrix((av, aj, ai), shape=A.getSize())


def solve_poisson_zero_neumann(msh, sources, polynomial_degree=1):
    """
    Solve Poisson equation with zero Neumann BC and point sources.
    
    -Δu = f  in Ω
    ∂u/∂n = 0  on ∂Ω
    """
    # Create function space
    V = functionspace(msh, ("Lagrange", polynomial_degree))
    
    # Create source term (regularized point sources)
    f = create_source_function(V, sources, sigma=0.03)
    
    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    
    # Bilinear form with small regularization for uniqueness
    epsilon = 1e-10
    a = inner(grad(u), grad(v)) * dx + epsilon * inner(u, v) * dx
    L = f * v * dx
    
    # Assemble system
    a_form = form(a)
    L_form = form(L)
    
    # Assemble matrix
    A_petsc = assemble_matrix(a_form)
    A_petsc.assemble()
    
    # Convert to scipy
    A = petsc_to_scipy(A_petsc)
    
    # Assemble RHS manually
    b = np.zeros(V.dofmap.index_map.size_local)
    
    # Simple assembly of RHS: b_i = integral(f * phi_i)
    # For P1 elements, this is approximately f_i * area/3 for each connected cell
    # But easier: just use f values directly scaled appropriately
    
    # Get mesh data for integration
    x_dofs = V.tabulate_dof_coordinates()
    
    # Approximate integral using lumped mass
    # For a proper assembly, we use the form directly
    from dolfinx.fem import assemble_vector as assemble_vec
    b_vec = Function(V)
    
    # Create a petsc vector and assemble
    from petsc4py import PETSc
    b_petsc = A_petsc.createVecRight()
    
    # Manual RHS assembly using form
    with b_petsc.localForm() as b_local:
        b_local.set(0.0)
    
    # Use dolfinx to assemble the vector
    from dolfinx.fem.petsc import assemble_vector as petsc_assemble_vector
    petsc_assemble_vector(b_petsc, L_form)
    b_petsc.assemble()
    
    # Convert to numpy
    b = b_petsc.getArray().copy()
    
    # Solve with scipy
    u_values = spsolve(A, b)
    
    # Create solution function
    u_h = Function(V)
    u_h.x.array[:] = u_values
    
    # Enforce mean-zero
    mean_u = assemble_scalar(form(u_h * dx))
    area = assemble_scalar(form(Constant(msh, default_scalar_type(1.0)) * dx))
    u_h.x.array[:] -= mean_u / area
    
    return u_h


def get_boundary_values(u, n_points=100):
    """Extract solution values on the boundary."""
    V = u.function_space
    
    # Get DOF coordinates
    dof_coords = V.tabulate_dof_coordinates()
    
    # Find boundary DOFs
    radii = np.sqrt(dof_coords[:, 0]**2 + dof_coords[:, 1]**2)
    boundary_mask = np.abs(radii - 1.0) < 0.1
    
    boundary_coords = dof_coords[boundary_mask]
    boundary_values = u.x.array[boundary_mask]
    
    angles = np.arctan2(boundary_coords[:, 1], boundary_coords[:, 0])
    sort_idx = np.argsort(angles)
    
    return angles[sort_idx], boundary_values[sort_idx]


def plot_solution(u, sources, save_path=None):
    """Plot the solution using matplotlib."""
    from matplotlib.tri import Triangulation
    
    msh = u.function_space.mesh
    coords = msh.geometry.x
    
    # Get connectivity
    msh.topology.create_connectivity(2, 0)
    cells = msh.topology.connectivity(2, 0)
    triangles = np.array([cells.links(i) for i in range(cells.num_nodes)])
    
    # Get solution values
    if len(u.x.array) == len(coords):
        u_values = u.x.array
    else:
        u_values = u.x.array[:len(coords)]
    
    tri = Triangulation(coords[:, 0], coords[:, 1], triangles)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    tcf = ax.tricontourf(tri, u_values, levels=50, cmap='viridis')
    plt.colorbar(tcf, ax=ax, label='Solution u')
    
    # Plot sources
    for (x, y), intensity in sources:
        color = 'red' if intensity > 0 else 'blue'
        marker = '+' if intensity > 0 else '*'
        ax.plot(x, y, marker, color=color, markersize=15, markeredgewidth=3)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title("Solution of Poisson's Equation with Zero Neumann BCs")
    ax.set_aspect('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_boundary_solution(angles, values, save_path=None):
    """Plot the solution values on the boundary."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(angles, values, 'o-', markersize=4)
    ax.set_xlabel('Angle (radians)')
    ax.set_ylabel('Solution Value')
    ax.set_title("Solution of Poisson's Equation on the Boundary")
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("Forward Solver for Poisson Equation with Point Sources")
    print("=" * 60)
    
    # Define sources (must sum to zero for Neumann problem)
    sources = [
        ((-0.2, 0.45), 1.0),
        ((0.6, 0.35), 1.0),
        ((-0.6, -0.3), -1.0),
        ((0.5, -0.1), -1.0),
        ((0.3, -0.65), -1.0),
        ((-0.15, 0.45), 1.0),
    ]
    
    print(f"\nNumber of sources: {len(sources)}")
    total = sum(q for _, q in sources)
    print(f"Total intensity: {total:.6f}")
    
    print("\nCreating mesh...")
    msh, _, _ = create_disk_mesh(radius=1.0, resolution=0.05)
    print(f"Mesh created with {msh.topology.index_map(0).size_local} vertices")
    
    print("\nSolving Poisson equation...")
    u_h = solve_poisson_zero_neumann(msh, sources)
    print("Solution computed!")
    
    print(f"\nSolution statistics:")
    print(f"  Min: {u_h.x.array.min():.6f}")
    print(f"  Max: {u_h.x.array.max():.6f}")
    print(f"  Mean: {u_h.x.array.mean():.6f}")
    
    print("\nExtracting boundary values...")
    angles, boundary_values = get_boundary_values(u_h)
    print(f"Extracted {len(angles)} boundary points")
    
    print("\nGenerating plots...")
    plot_solution(u_h, sources, save_path="solution_disk.png")
    plot_boundary_solution(angles, boundary_values, save_path="solution_boundary.png")
    
    print("\nDone!")
