import collections
import numpy as np

_DIM_MISMATCH_MSG = 'The number of rows of the target matrix to segment ' \
                    'is different from the number of labels.'

def ranks(a):
    """
    Get the ranking of each element in a and
    replace the items with their ranks.
    :param a: target list
    :return: list whose items are replaced by ranks
    """
    # Sort values in a without repetition.
    a_sorted = sorted(list(set(a)))

    # Create a dictionary to map the values and their rankings.
    ranking_dict = {a_sorted[i]: i for i in range(len(a_sorted))}

    # Replace the values with their rankings.
    a_replaced = [ranking_dict[a[i]] for i in range(len(a))]
    return a_replaced


def segment(X, y):
    """
    Arrange row vectors in X according to labels y,
    such that rows with the same label will be put together.
    :param X: target matrix to segment
    :param y: labels used for segmenting the target
    :return: segmented target matrix as a list of matrices
    """
    if len(y) != X.shape[0]:
        raise RuntimeError(_DIM_MISMATCH_MSG)

    # Count number of classes in y.
    num_classes = len(set(y))

    # Map the labels to the range [0, num_classes - 1].
    y_rank = ranks(y)

    # Initialize matrices for each class.
    X_seg = [[] for _ in range(num_classes)]

    # Put the target matrix into segments.
    for i in range(len(y_rank)):
        cls_label = y_rank[i]
        X_seg[cls_label].append(X[i, :])

    # Merge the matrices for all classes.
    X_seg = [np.vstack(X_seg_i) for X_seg_i in X_seg]
    return X_seg


def unsegment(X_seg, y):
    """
    Reorder the segmented matrix into its original order.
    :param X_seg: segmented matrix
    :param y: labels used for segmenting the original matrix
    :return: unsegmented matrix
    """
    if X_seg.shape[0] != len(y):
        raise RuntimeError(_DIM_MISMATCH_MSG)

    # Initialize unsegmented matrix.
    X = []

    # Map the labels to the range [0, num_classes - 1].
    y_rank = ranks(y)

    # Obtain the number of occurrences for each class label.
    cls_counts = collections.Counter(y_rank).items()
    cls_counts = sorted(cls_counts, key=lambda item: item[0])

    # Initialize pointers for placing the row vectors back.
    p = {}

    # Zero the counter for number of class labels already counted.
    # This will be used as the starting point of the pointers.
    labels_counter = 0
    for label, count in cls_counts:
        p[label] = labels_counter
        labels_counter += count

    # Reorder the matrix.
    for i in range(len(y_rank)):

        # Find the segmented position for i-th row in original matrix.
        segmented_position = p[y_rank[i]]

        # Add the row vector in the original order.
        X.append(X_seg[segmented_position])

        # Increment the pointer for this class.
        p[y_rank[i]] += 1
    return np.vstack(X) if len(X) > 1 else np.array(X)


class LNMF:
    def __init__(
        self, 
        n_components=None, 
        alpha=0.1, 
        beta=0.1, 
        gamma=0.1, 
        max_iters=1000
    ):
        """
        Initialise the Labeled Non-negative Matrix Factorisation algorithm.
        :param n_components: latent dimension for the factor matrices
        :param alpha: H regularization coefficient
        :param beta: label modulation coefficient
        :param gamma: W regularization coefficient
        :param max_iters: Maximum number of iterations for the gradient descent
        """
        self.n_components = n_components
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.components_ = None  # H matrix in V = W @ H
        self.max_iters = max_iters

    def _grad_desc_h(self, V, W, H):
        """
        Perform gradient descent on H.
        """

        # Keep the original H value.
        H_0 = H.copy()
        WV = W.T.dot(V)
        WWH = W.T.dot(W.dot(H))
        eta = -np.divide(H, WWH)
        H = np.multiply(eta, -WV + self.beta * H)
        H = np.nan_to_num(H)

        # Check whether H has converged.
        return H, np.linalg.norm(H_0 - H) == 0

    def _grad_desc_w(self, V_seg, W_seg, H, num_classes):
        """
        Perform gradient descent on W.
        """

        # Keep the original W value.
        W = np.vstack(W_seg)
        W_0 = W.copy()

        # Update matrix for each class.
        for class_i in range(num_classes):
            W_i = W_seg[class_i]
            V_i = V_seg[class_i]
            n_i = W_seg[class_i].shape[0]

            # Calculate the learning rate.
            WHH_i = W_i.dot(H.dot(H.T))
            eta_i = -np.divide(W_i, WHH_i)

            # Calculate the reconstruction term.
            recon_term = -V_i.dot(H.T)

            # Calculate the label modulation term.
            W_i_sum = np.sum(W_i, axis=0)
            C = np.vstack([W_i_sum] * n_i)
            mod_term = self.alpha * (n_i * W_i - C)

            # Calculate the regularisation term.
            reg_term = self.gamma * W_i

            # Sum up the gradient.
            gradient = recon_term + mod_term + reg_term

            # Gradient descent step.
            W_seg[class_i] = np.multiply(eta_i, gradient)

            # Ensure non-negativity of the factor matrix.
            W_seg[class_i] = np.nan_to_num(W_seg[class_i])
            W_seg[class_i][W_seg[class_i] < 0] = 0

        # Check whether W has converged.
        return W_seg, np.linalg.norm(W_0 - W) == 0

    def fit_transform(self, X, y, W=None, H=None):
        """
        Labeled Non-negative Matrix Factorization with regularization.
        :param X: target matrix
        :param y: class labels
        :param W: initial guess for W
        :param H: initial guess for H
        :return: factor matrices
        """
        if (X < 0).any():
            raise RuntimeError('Target matrix is not non-negative.')
        n, m = X.shape

        # Count the number of classes.
        num_classes = len(set(y))

        # Define the latent dimension.
        r = self.n_components if self.n_components > 0 else min(n, m)

        # Randomly initialise the factor matrices if initial values not given.
        W = W if W is not None else np.abs(np.random.normal(size=(n, r)))
        H = H if H is not None else np.abs(np.random.normal(size=(r, m)))

        # Segmentation on the target, factor and label matrix.
        V_seg = segment(X, y)
        V = np.vstack(V_seg)
        W_seg = segment(W, y)
        W = np.vstack(W_seg)

        # Iterations.
        for _ in range(self.max_iters):

            # Update H.
            H, H_converged = self._grad_desc_h(V, W, H)

            # Update W.
            W_seg, W_converged = self._grad_desc_w(
                V_seg, W_seg, H, num_classes)
            W = np.vstack(W_seg)

            # Check for convergence.
            if H_converged and W_converged:
                break

        # Undo the segmentation.
        W = unsegment(np.vstack(W_seg), y)
        self.components_ = H
        return W
