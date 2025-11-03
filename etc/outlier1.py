import numpy as np

def load_data(csv_file):

    # 1. Load the data
    data = np.genfromtxt(csv_file, delimiter=',', dtype=str)
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
    return data

def simple_pca_2d(data):

    # 2. Compute Gram Matrix (no centering)
    gram_matrix = np.dot(data.T, data)

    # 3. Compute Eigenvalues and Eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(gram_matrix)

    # 4. Select the Top 2 Eigenvectors (Principal Components)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    top2_indices = sorted_indices[:2]
    principal_components = eigenvectors[:, top2_indices]

    # Project data
    data_2d = np.dot(data, principal_components)
    return data_2d

def find_most_influential_point(original_csv, pca_2d_csv):
    data = load_data(original_csv)
    reference_2d = load_data(pca_2d_csv)

    n_samples = data.shape[0]
    max_diff = -1
    influential_index = -1

    for i in range(n_samples):
        # Leave out i-th row
        data_reduced = np.delete(data, i, axis=0)

        # Perform PCA on reduced data
        projected_reduced = simple_pca_2d(data_reduced)

        # Compare with reference 2D (skip the removed point)
        reference_reduced = np.delete(reference_2d, i, axis=0)

        # Sum of squared differences
        diff = np.sum((projected_reduced - reference_reduced) ** 2)

        if diff > max_diff:
            max_diff = diff
            influential_index = i

    print(f"Most influential data point is at index {influential_index} with difference {max_diff}")
    return influential_index

# Usage
influential_idx = find_most_influential_point('Iris.csv', 'test11.csv')
