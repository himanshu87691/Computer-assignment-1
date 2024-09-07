import numpy as np

# Define the size of the matrix
n = 100

# Initialize the main diagonal, super-diagonal, and sub-diagonal
initial_guess = np.ones(n)
# Initialize the RHS vector
rhs = np.zeros(n)
rhs[0] = 1

def create_tridiagonal_matrix(n):
    # Create an n x n matrix of zeros
    matrix = np.zeros((n, n))
    
    # Set the main diagonal to 2.0004
    np.fill_diagonal(matrix, 2.0004)
    
    # Set the super-diagonal to -1
    np.fill_diagonal(matrix[1:], -1)
    
    # Set the sub-diagonal to -1
    np.fill_diagonal(matrix[:, 1:], -1)
    
    # Adjust the last sub-diagonal element to -2
    matrix[-1, -2] = -2
    
    return matrix

# Create a 100x100 tridiagonal matrix
A = create_tridiagonal_matrix(n)
b = rhs
x0 = initial_guess

def gauss_seidel(A, b, x0, tol=0.0001, max_iterations=2498):
    """
    Solves the system of linear equations Ax = b using the Gauss-Seidel method.

    Parameters:
    A : 2D array
        Coefficient matrix.
    b : 1D array
        Right-hand side vector.
    x0 : 1D array
        Initial guess for the solution.
    tol : float
        Tolerance for convergence (default is 1e-4).
    max_iterations : int
        Maximum number of iterations (default is 100).

    Returns:
    x : 1D array
        Solution vector.
    """

    l = len(b)
    x = x0.copy()

    for iteration in range(max_iterations):
        x_old = x.copy()

        for i in range(l):
            sum1 = sum(A[i][j] * x[j] for j in range(i))
            sum2 = sum(A[i][j] * x_old[j] for j in range(i + 1, l))
            x[i] = (b[i] - sum1 - sum2) / A[i][i]

        # Check for convergence
        if np.linalg.norm(x - x_old, ord=np.inf) < tol:
            print(f"Converged in {iteration + 1} iterations.")
            return x

    print("Maximum iterations reached without convergence.")
    return x

# Solve the system
solution = gauss_seidel(A, b, x0)
print("Solution:", solution)
