import numpy as np
import logging
import pdb

class KMeans:
    """
    K-Means clustering algorithm using L2 (Euclidean) distance.

    Parameters:
    -----------
    n_clusters : int
        The number of clusters to form as well as the number of centroids to generate.
    max_iter : int, default=100
        Maximum number of iterations of the k-medians algorithm for a single run.
    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare convergence.
    random_state : int, default=None
        Determines random number generation for centroid initialization.
    verbose : bool, default=False
        Verbosity mode.
    """
    def __init__(self, n_clusters, max_iter=2000, tol=1e-10, random_state=None, verbose=False):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        self.centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0

    def _l2_distance(self, X, Y):
        """Compute L2 (Euclidean) distance between each pair of the two collections of points."""
        return np.sum(np.abs(X[:, np.newaxis, :] - Y) ** 2, axis=2)

    def _initialize_centers(self, X):
        """Initialize cluster centers using k-means++ algorithm."""
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples = X.shape[0]
        centers = np.zeros((self.n_clusters, X.shape[1]))

        # Choose first center randomly
        first_idx = np.random.randint(n_samples)
        centers[0] = X[first_idx]

        # Initialize distances
        distances = np.full(n_samples, np.inf)

        # Choose remaining centers
        for i in range(1, self.n_clusters):
            # Compute distances to the nearest center for each point
            distances = np.minimum(distances, np.sum(np.abs(X - centers[i-1]), axis=1))

            # Choose new center with probability proportional to distance^2
            probs = distances ** 2
            probs /= probs.sum()

            # Ensure we don't select duplicate centers
            while True:
                idx = np.random.choice(n_samples, p=probs)
                if not np.any(np.all(X[idx] == centers[:i], axis=1)):
                    break

            centers[i] = X[idx]

        return centers

    def fit(self, X, y=None):
        """
        Compute k-means clustering.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.
        y : Ignored
            Not used, present here for API consistency.

        Returns:
        --------
        self : object
            Fitted estimator.
        """
        X = np.asarray(X)
        n_samples, n_features = X.shape

        # Initialize cluster centers
        centers = self._initialize_centers(X)

        # Initialize variables
        labels = np.zeros(n_samples, dtype=int)
        prev_inertia = None

        # K-means algorithm
        for i in range(self.max_iter):
            # Assign points to nearest center (L2 distance)
            distances = self._l2_distance(X, centers)
            labels = np.argmin(distances, axis=1)

            # Update centers to be the mean of points in each cluster
            for j in range(self.n_clusters):
                mask = (labels == j)
                if np.any(mask):
                    centers[j] = np.mean(X[mask], axis=0)

            # Calculate inertia (sum of L2 distances to nearest center)
            inertia = np.sum(np.min(distances, axis=1))
            logging.info(f"Iteration {i+1}, Total L2 distances to centers: {inertia}")

            # Check for convergence
            if prev_inertia is not None and np.abs(prev_inertia - inertia) < self.tol * inertia:
                if self.verbose:
                    logging.info(f"Converged at iteration {i+1}")
                break

            prev_inertia = inertia

        self.centers_ = centers
        self.labels_ = labels
        self.inertia_ = prev_inertia
        self.n_iter_ = i + 1

        return self

    def predict(self, X):
        """
        Predict the closest cluster each sample in X belongs to.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            New data to predict.

        Returns:
        --------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        X = np.asarray(X)
        distances = self._l2_distance(X, self.centers_)
        return np.argmin(distances, axis=1)

    def fit_predict(self, X, y=None):
        """
        Compute cluster centers and predict cluster index for each sample.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            New data to transform.
        y : array-like of shape (n_samples,), default=None
            Binary mask where 1 indicates samples to be processed.
            If None, all samples are processed.

        Returns:
        --------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
            Samples where y == 0 will have label -1.
        """
        # If y is not provided, process all samples
        if y is None:
            return self.fit(X).labels_

        y = np.asarray(y, dtype=bool)
        if len(y) != len(X):
            raise ValueError(f"Length of y ({len(y)}) does not match length of X ({len(X)})")

        # Initialize labels with -1 (for y == 0 samples)
        labels = np.full(len(X), -1, dtype=int)

        # If there are samples to process (y == 1)
        if np.any(y):
            # Fit and predict only on samples where y == 1
            self.fit(X[y])
            labels[y] = self.labels_

        return labels