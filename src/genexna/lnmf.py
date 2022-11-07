"""
TODO
"""

import collections
import numpy as np

_DIM_MISMATCH_MSG = 'The number of rows of the target matrix to segment ' \
                    'is different from the number of labels.'


def ranks(mat):
    """
    Get the ranking of each element in a and
    replace the items with their ranks.
    :param a: target list
    :return: list whose items are replaced by ranks
    """
    # Sort values in a without repetition.
    a_sorted = sorted(list(set(mat)))

    # Create a dictionary to map the values and their rankings.
    ranking_dict = {a_sorted[i]: i for i in range(len(a_sorted))}

    # Replace the values with their rankings.
    a_replaced = [ranking_dict[mat[i]] for i in range(len(mat))]
    return a_replaced


def segment(mat, labels):
    """
    Arrange row vectors in X according to labels y,
    such that rows with the same label will be put together.
    :param mat: target matrix to segment
    :param y: labels used for segmenting the target
    :return: segmented target matrix as a list of matrices
    """
    if len(labels) != mat.shape[0]:
        raise RuntimeError(_DIM_MISMATCH_MSG)

    # Count number of classes in y.
    num_classes = len(set(labels))

    # Map the labels to the range [0, num_classes - 1].
    label_ranks = ranks(labels)

    # Initialize matrices for each class.
    mat_seg = [[] for _ in range(num_classes)]

    # Put the target matrix into segments.
    for i, cls_label in enumerate(label_ranks):
        mat_seg[cls_label].append(mat[i, :])

    # Merge the matrices for all classes.
    mat_seg = [np.vstack(X_seg_i) for X_seg_i in mat_seg]
    return mat_seg


def unsegment(mat_seg, labels):
    """
    Reorder the segmented matrix into its original order.
    :param X_seg: segmented matrix
    :param y: labels used for segmenting the original matrix
    :return: unsegmented matrix
    """
    if mat_seg.shape[0] != len(labels):
        raise RuntimeError(_DIM_MISMATCH_MSG)

    # Initialize unsegmented matrix.
    mat = []

    # Map the labels to the range [0, num_classes - 1].
    label_ranks = ranks(labels)

    # Obtain the number of occurrences for each class label.
    cls_counts = collections.Counter(label_ranks).items()
    cls_counts = sorted(cls_counts, key=lambda item: item[0])

    # Initialize pointers for placing the row vectors back.
    ptr = {}

    # Zero the counter for number of class labels already counted.
    # This will be used as the starting point of the pointers.
    labels_counter = 0
    for label, count in cls_counts:
        ptr[label] = labels_counter
        labels_counter += count

    # Reorder the matrix.
    for label_rank in label_ranks:

        # Find the segmented position for i-th row in original matrix.
        segmented_position = ptr[label_rank]

        # Add the row vector in the original order.
        mat.append(mat_seg[segmented_position])

        # Increment the pointer for this class.
        ptr[label_rank] += 1
    return np.vstack(mat) if len(mat) > 1 else np.array(mat)


def _grad_desc_h(v, w, h, beta):
    """
    Perform gradient descent on H.
    """

    # Keep the original H value.
    h_0 = h.copy()
    w_v = w.T.dot(v)
    w_w_h = w.T.dot(w.dot(h))
    eta = -np.divide(h, w_w_h)
    h = np.multiply(eta, -w_v + beta * h)
    h = np.nan_to_num(h)

    # Check whether H has converged.
    return h, np.linalg.norm(h_0 - h) == 0


def _grad_desc_w(v_seg, w_seg, h, num_classes, alpha, gamma):
    """
    Perform gradient descent on W.
    """

    # Keep the original W value.
    w = np.vstack(w_seg)
    w_0 = w.copy()

    # Update matrix for each class.
    for class_i in range(num_classes):
        w_i = w_seg[class_i]
        v_i = v_seg[class_i]
        n_i = w_seg[class_i].shape[0]

        # Calculate the learning rate.
        w_h_h_i = w_i.dot(h.dot(h.T))
        eta_i = -np.divide(w_i, w_h_h_i)

        # Calculate the reconstruction term.
        recon_term = -v_i.dot(h.T)

        # Calculate the label modulation term.
        w_i_sum = np.sum(w_i, axis=0)
        c = np.vstack([w_i_sum] * n_i)
        mod_term = alpha * (n_i * w_i - c)

        # Calculate the regularisation term.
        reg_term = gamma * w_i

        # Sum up the gradient.
        gradient = recon_term + mod_term + reg_term

        # Gradient descent step.
        w_seg[class_i] = np.multiply(eta_i, gradient)

        # Ensure non-negativity of the factor matrix.
        w_seg[class_i] = np.nan_to_num(w_seg[class_i])
        w_seg[class_i][w_seg[class_i] < 0] = 0

    # Check whether W has converged.
    return w_seg, np.linalg.norm(w_0 - w) == 0


def factorize(mat, labels, w_init=None, h_init=None,
              n_components=None,
              alpha=0.1,
              beta=0.1,
              gamma=0.1,
              max_iters=1000,
              return_w=False):
    """
    Labeled Non-negative Matrix Factorization with regularization.
    :param mat: target matrix
    :param labels: class labels
    :param w_init: initial guess for W
    :param h_init: initial guess for H
    :param n_components: latent dimension for the factor matrices
    :param alpha: H regularization coefficient
    :param beta: label modulation coefficient
    :param gamma: W regularization coefficient
    :param max_iters: Maximum number of iterations for the gradient descent
    :return: factor matrices
    """
    if (mat < 0).any():
        raise RuntimeError('Target matrix is not non-negative.')
    n, m = mat.shape

    # Count the number of classes.
    num_classes = len(set(labels))

    # Define the latent dimension.
    r = n_components if n_components > 0 else min(n, m)

    # Randomly initialise the factor matrices if initial values not given.
    w = w_init if w_init is not None else np.abs(np.random.normal(size=(n, r)))
    h = h_init if h_init is not None else np.abs(np.random.normal(size=(r, m)))

    # Segmentation on the target, factor and label matrix.
    v_seg = segment(mat, labels)
    v = np.vstack(v_seg)
    w_seg = segment(w, labels)
    w = np.vstack(w_seg)

    # Iterations.
    for _ in range(max_iters):

        # Update H.
        h, h_converged = _grad_desc_h(v, w, h, beta)

        # Update W.
        w_seg, w_converged = _grad_desc_w(
            v_seg, w_seg, h, num_classes, alpha, gamma)
        w = np.vstack(w_seg)

        # Check for convergence.
        if h_converged and w_converged:
            break
    w = unsegment(np.vstack(w_seg), labels)

    return h if not return_w else (w, h)
