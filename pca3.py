import numpy as np
import sys

def simple_pca(input_filename: str, output_filename: str):
    """
    Performs a simple PCA to reduce dimensionality to 2 principal components.

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

        # Normalize rows by dividing by their Euclidean norm
        row_norms = np.linalg.norm(data, axis=1, keepdims=True)
        row_norms[row_norms == 0] = 1  # Avoid division by zero
        data = data / row_norms

        # 2. Mean-center the data
        mean_vector = np.mean(data, axis=0)
        data_centered = data - mean_vector

        # 3. Compute Gram matrix X^T X
        gram_matrix = np.dot(data_centered.T, data_centered)

        # 4. Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(gram_matrix)

        # 5. Select top 2 principal components
        top_indices = np.argsort(eigenvalues)[::-1][:2]  # descending order
        principal_components = eigenvectors[:, top_indices]  # m x 2

        # 6. Project the data
        data_projected = np.dot(data_centered, principal_components)  # n x 2

        # 7. Save the result
        np.savetxt(output_filename, data_projected, delimiter=',')

        print(f"PCA complete. Output saved to '{output_filename}'.")

    except Exception as e:
        print(f"Error during PCA: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python simple_pca.py <input_filename> <output_filename>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    simple_pca(input_file, output_file)
