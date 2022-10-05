def vector_distance_batch(vector_1, vectors_all):
    """
    Return poincare distances between one vector and a set of other vectors.
    Parameters
    ----------
    vector_1 : numpy.array
        vector from which Poincare distances are to be computed.
        expected shape (dim,)
    vectors_all : numpy.array
        for each row in vectors_all, distance from vector_1 is computed.
        expected shape (num_vectors, dim)
    Returns
    -------
    numpy.array
        Contains Poincare distance between vector_1 and each row in vectors_all.
        shape (num_vectors,)
    """
    euclidean_dists = np.linalg.norm(vector_1 - vectors_all, axis=1)
    norm = np.linalg.norm(vector_1)
    all_norms = np.linalg.norm(vectors_all, axis=1)
    return np.arccosh(
        1 + 2 * (
            (euclidean_dists ** 2) / ((1 - norm ** 2) * (1 - all_norms ** 2))
        )
    )

def vector_distance(vector_1, vector_2):
    """
    Return poincare distance between two input vectors. Convenience method over `vector_distance_batch`.
    Parameters
    ----------
    vector_1 : numpy.array
        input vector
    vector_2 : numpy.array
        input vector
    Returns
    -------
    numpy.float
        Poincare distance between `vector_1` and `vector_2`.
    """
    return PoincareKeyedVectors.vector_distance_batch(vector_1, vector_2[np.newaxis, :])[0]