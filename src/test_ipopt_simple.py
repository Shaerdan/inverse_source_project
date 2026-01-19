"""Minimal test to verify IPOPT constraint enforcement."""
import numpy as np

try:
    import cyipopt
except ImportError:
    print("cyipopt not available")
    exit(1)

class SimpleDiskProblem:
    """Minimize x² + y² subject to x² + y² <= 0.5²"""
    
    def __init__(self):
        self.n_vars = 2
        self.n_constraints = 1
    
    def objective(self, x):
        # Minimize (x-0.3)² + (y-0.3)² (minimum at (0.3, 0.3) which is inside disk)
        return (x[0] - 0.3)**2 + (x[1] - 0.3)**2
    
    def gradient(self, x):
        return np.array([2*(x[0] - 0.3), 2*(x[1] - 0.3)])
    
    def constraints(self, x):
        # c(x) = 0.5² - x² - y² >= 0
        return np.array([0.25 - x[0]**2 - x[1]**2])
    
    def jacobian(self, x):
        # dc/dx = -2x, dc/dy = -2y
        return np.array([-2*x[0], -2*x[1]])
    
    def jacobianstructure(self):
        return (np.array([0, 0]), np.array([0, 1]))

# Setup
problem = SimpleDiskProblem()

nlp = cyipopt.Problem(
    n=2,
    m=1,
    problem_obj=problem,
    lb=np.array([-1.0, -1.0]),
    ub=np.array([1.0, 1.0]),
    cl=np.array([0.0]),
    cu=np.array([1e20])
)

nlp.add_option('print_level', 5)
nlp.add_option('hessian_approximation', 'limited-memory')

# Start OUTSIDE the feasible region
x0 = np.array([0.8, 0.8])  # r = 1.13 > 0.5
print(f"Initial point: {x0}, r = {np.sqrt(x0[0]**2 + x0[1]**2):.3f}")
print(f"Initial constraint value: {0.25 - x0[0]**2 - x0[1]**2:.3f} (should be >= 0)")

x_opt, info = nlp.solve(x0)

print(f"\nFinal point: {x_opt}, r = {np.sqrt(x_opt[0]**2 + x_opt[1]**2):.3f}")
print(f"Final constraint value: {0.25 - x_opt[0]**2 - x_opt[1]**2:.3f}")
print(f"Status: {info['status_msg']}")
