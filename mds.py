import sys
import numpy as np
from numpy.linalg import eigh

def simple_mds(input_filename: str, output_filename: str, alpha: float):
    """
    Performs Classical Multidimensional Scaling (MDS) to reduce dimensionality to 2 components.
    
    Args:
        input_filename (str): Path to the input CSV file (n x m matrix).
        output_filename (str): Path to the output CSV file (n x 2 matrix).
        alpha (float): Exponential scaler of the distance matrix (>=0).
    """
    # 1. Load data
    data = np.genfromtxt(input_filename, delimiter=",", dtype=str)
    
    # Convert categorical columns (non-numeric) to integers
    for col in range(data.shape[1]):
        try:
            data[:, col] = data[:, col].astype(float)
        except ValueError:
            # Encode categorical strings
            unique_vals = {val: i for i, val in enumerate(np.unique(data[:, col]))}
            data[:, col] = np.array([unique_vals[val] for val in data[:, col]])
    data = data.astype(float)
    
    n_samples = data.shape[0]
    
    # 2. Compute Squared Euclidean Distance Matrix
    dist_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i, n_samples):
            dist_sq = np.sum((data[i] - data[j])**2)
            dist_scaled = dist_sq**(alpha / 2)
            dist_matrix[i, j] = dist_matrix[j, i] = dist_scaled
    
    # 3. Double Centering to get Gram matrix
    row_mean = np.mean(dist_matrix, axis=1, keepdims=True)
    col_mean = np.mean(dist_matrix, axis=0, keepdims=True)
    total_mean = np.mean(dist_matrix)
    G = -0.5 * (dist_matrix - row_mean - col_mean + total_mean)
    
    # 4. Eigen decomposition of Gram matrix
    eigenvalues, eigenvectors = eigh(G)
    
    # 5. Select top 2 eigenvectors corresponding to largest eigenvalues
    idx = np.argsort(eigenvalues)[::-1][:2]
    top_eigenvalues = eigenvalues[idx]
    top_eigenvectors = eigenvectors[:, idx]
    
    # 6. Compute projected coordinates
    Lambda_sqrt = np.sqrt(np.diag(top_eigenvalues))
    data_projected = top_eigenvectors @ Lambda_sqrt
    
    # 7. Save to CSV
    np.savetxt(output_filename, data_projected, delimiter=",", fmt="%.6f")

# --- Run from command line ---
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python simple_mds.py <input_filename> <output_filename> <alpha>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    alpha_val = float(sys.argv[3])
    
    simple_mds(input_file, output_file, alpha_val)
