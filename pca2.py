import numpy as np
import sys
import csv

def simple_pca(input_filename: str, output_filename: str):
    """
    Performs a simple PCA to reduce dimensionality to 2 principal components.
    Categorical columns are encoded as integers.
    """
    try:
        # 1. Load the data
        data_list = []
        with open(input_filename, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                data_list.append(row)
        data_list = np.array(data_list, dtype=object)  # keep as object for mixed types

        # Handle single-row edge case
        if data_list.ndim == 1:
            data_list = data_list.reshape(1, -1)

        # Encode categorical columns
        n_rows, n_cols = data_list.shape
        data_numeric = np.zeros((n_rows, n_cols))
        for col in range(n_cols):
            try:
                # Try converting to float
                data_numeric[:, col] = data_list[:, col].astype(float)
            except ValueError:
                # Encode as integers if not numeric
                unique_vals = {v: i for i, v in enumerate(np.unique(data_list[:, col]))}
                data_numeric[:, col] = np.array([unique_vals[v] for v in data_list[:, col]])
        print(data_numeric)

        # 2. Standardize the data (Mean-Centering)
        mean_vals = np.mean(data_numeric, axis=0)
        data_centered = data_numeric - mean_vals

        # 3. Compute covariance via Gram matrix
        gram_matrix = np.dot(data_centered.T, data_centered)

        # 4. Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(gram_matrix)  # eigh is for symmetric matrices

        # 5. Select the Top 2 Principal Components
        sorted_indices = np.argsort(eigenvalues)[::-1]  # descending order
        top_indices = sorted_indices[:2]
        principal_components = eigenvectors[:, top_indices]  # m x 2

        # 6. Project the Data
        data_projected = np.dot(data_centered, principal_components)  # n x 2

        # 7. Save the Result
        np.savetxt(output_filename, data_projected, delimiter=',', fmt='%.6f')
        print(f"PCA projection saved to {output_filename}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py input.csv output.csv")
    else:
        simple_pca(sys.argv[1], sys.argv[2])
