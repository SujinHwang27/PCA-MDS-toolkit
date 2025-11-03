import sys
import numpy as np
import csv

def simple_pca(input_filename: str, output_filename: str):
    """
    Performs a simple PCA to reduce dimensionality to 2 principal components.

    The input is a comma-separated matrix (CSV file) of size n x m.
    The output is a comma-separated matrix (CSV file) of size n x 2,
    representing the data projected onto the first two principal components.

    Args:
        input_filename (str): Path to the input CSV file (n x m matrix).
        output_filename (str): Path to the output CSV file (n x 2 matrix).
    """
    try:
        # 1. Load the data
        data = np.genfromtxt(input_filename, delimiter=',', dtype=str)
        # Try converting to float; if fails, encode non-numeric features
        try:
            data = data.astype(float)
        except ValueError:
            # Encode categorical columns
            for j in range(data.shape[1]):
                try:
                    data[:, j] = data[:, j].astype(float)
                except ValueError:
                    unique_vals = {v: i for i, v in enumerate(np.unique(data[:, j]))}
                    data[:, j] = np.vectorize(unique_vals.get)(data[:, j])
            data = data.astype(float)

        # Handle single-row edge case
        if data.ndim == 1:
            data = data.reshape(1, -1)

        # 2. Compute Gram Matrix (no centering)
        gram_matrix = np.dot(data.T, data)

        # 3. Compute Eigenvalues and Eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(gram_matrix)

        # 4. Select the Top 2 Eigenvectors (Principal Components)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        top2_indices = sorted_indices[:2]
        principal_components = eigenvectors[:, top2_indices]

        # 5. Project the Data onto the 2 Principal Components
        data_projected = np.dot(data, principal_components)

        # 6. Save the Result
        np.savetxt(output_filename, data_projected, delimiter=',')

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python simple_pca.py <input_filename> <output_filename>")
        sys.exit(1)
    simple_pca(sys.argv[1], sys.argv[2])
