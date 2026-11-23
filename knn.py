"""
Implements a K-Nearest Neighbor classifier in PyTorch.
"""
import torch
from typing import Dict, List


def hello():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up.
    """
    print("Hello from knn.py!")


def compute_distances_two_loops(x_train: torch.Tensor, x_test: torch.Tensor):
    """
    Computes the squared Euclidean distance between each element of training
    set and each element of test set. Images should be flattened and treated
    as vectors.

    This implementation uses a naive set of nested loops over the training and
    test data.

    The input data may have any number of dimensions -- for example this
    function should be able to compute nearest neighbor between vectors, in
    which case the inputs will have shape (num_{train, test}, D); it should
    also be able to compute nearest neighbors between images, where the inputs
    will have shape (num_{train, test}, C, H, W). More generally, the inputs
    will have shape (num_{train, test}, D1, D2, ..., Dn); we flatten
    each element of shape (D1, D2, ..., Dn) into a vector of shape
    (D1 * D2 * ... * Dn) before computing distances.

    The input tensors are not modified.

    Args:
        x_train: Tensor of shape (num_train, D1, D2, ...)
        x_test: Tensor of shape (num_test, D1, D2, ...)

    Returns:
        dists: Tensor of shape (num_train, num_test) where dists[i, j]
            is the squared Euclidean distance between the i-th training point
            and the j-th test point. It has the same dtype as x_train.
    """
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    dists = x_train.new_zeros(num_train, num_test)
    # Flatten every sample into a vector (views, inputs are not modified).
    train_flat = x_train.reshape(num_train, -1)
    test_flat = x_test.reshape(num_test, -1)
    for i in range(num_train):
        for j in range(num_test):
            diff = train_flat[i] - test_flat[j]
            dists[i, j] = (diff * diff).sum()
    return dists


def compute_distances_one_loop(x_train: torch.Tensor, x_test: torch.Tensor):
    """
    Computes the squared Euclidean distance between each element of training
    set and each element of test set. Images should be flattened and treated
    as vectors.

    This implementation uses only a single loop over the training data.

    Similar to `compute_distances_two_loops`, this handles inputs with any
    number of dimensions. The inputs are not modified.

    Args:
        x_train: Tensor of shape (num_train, D1, D2, ...)
        x_test: Tensor of shape (num_test, D1, D2, ...)

    Returns:
        dists: Tensor of shape (num_train, num_test) where dists[i, j]
            is the squared Euclidean distance between the i-th training point
            and the j-th test point. It has the same dtype as x_train.
    """
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    dists = x_train.new_zeros(num_train, num_test)
    train_flat = x_train.reshape(num_train, -1)
    test_flat = x_test.reshape(num_test, -1)
    for i in range(num_train):
        # (num_test, D) - (D,) broadcasts over all test samples at once.
        diff = test_flat - train_flat[i]
        dists[i] = (diff * diff).sum(dim=1)
    return dists


def compute_distances_no_loops(x_train: torch.Tensor, x_test: torch.Tensor):
    """
    Computes the squared Euclidean distance between each element of training
    set and each element of test set. Images should be flattened and treated
    as vectors.

    This implementation uses no Python loops and does not create any
    intermediate tensor with O(num_train * num_test * D) elements. It uses

        ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b

    so that only a single (num_train, num_test) matrix is formed by a matrix
    multiply and two broadcast sums.

    Args:
        x_train: Tensor of shape (num_train, C, H, W)
        x_test: Tensor of shape (num_test, C, H, W)

    Returns:
        dists: Tensor of shape (num_train, num_test) where dists[i, j] is
            the squared Euclidean distance between the i-th training point and
            the j-th test point.
    """
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    train_flat = x_train.reshape(num_train, -1)
    test_flat = x_test.reshape(num_test, -1)

    train_sq = (train_flat * train_flat).sum(dim=1, keepdim=True)  # (N, 1)
    test_sq = (test_flat * test_flat).sum(dim=1, keepdim=True)  # (M, 1)

    # The only (N, M) tensor is the output itself; everything else is done
    # in place on it (one matrix multiply + two broadcast sums).
    dists = train_flat.mm(test_flat.t())
    dists.mul_(-2.0).add_(train_sq).add_(test_sq.t())
    # Floating point error can produce tiny negative values; clamp them.
    dists.clamp_(min=0)
    return dists
