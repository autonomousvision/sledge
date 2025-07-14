# flake8: noqa
# TODO: Fix this file and remove the noqa comment.
import gc
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
from tqdm import trange


@dataclass
class Manifold:
    features: npt.NDArray[np.float32]
    radii: npt.NDArray[np.float32]


def run_improved_precision_recall(
    features_r: npt.NDArray[np.float32], features_g: npt.NDArray[np.float32], k: int = 3
) -> Tuple[float, float]:

    max_size = 25_000
    features_r = features_r[:max_size]
    features_g = features_g[:max_size]

    distances_r = _compute_pairwise_distances(features_r)
    radii_r = _distances2radii(distances_r, k=k)
    manifold_r = Manifold(features_r, radii_r)

    distances_g = _compute_pairwise_distances(features_g)
    radii_g = _distances2radii(distances_g, k=k)
    manifold_g = Manifold(features_g, radii_g)

    precision = compute_metric(manifold_r, features_g)
    recall = compute_metric(manifold_g, features_r)

    return precision, recall


def compute_metric(manifold_ref: Manifold, feats_subject: npt.NDArray[np.float32], desc=""):
    num_subjects = feats_subject.shape[0]
    count = 0
    dist = _compute_pairwise_distances(manifold_ref.features, feats_subject)
    for i in trange(num_subjects, desc=desc):
        count += (dist[:, i] < manifold_ref.radii).any()
    return count / num_subjects


def get_kth_value(np_array, k):
    kprime = k + 1  # kth NN should be (k+1)th because closest one is itself
    idx = np.argpartition(np_array, kprime)
    k_smallests = np_array[idx[:kprime]]
    kth_value = k_smallests.max()
    return kth_value


def _distances2radii(distances, k=3):
    num_features = distances.shape[0]
    radii = np.zeros(num_features)
    for i in range(num_features):
        radii[i] = get_kth_value(distances[i], k=k)
    return radii


def is_in_ball(center, radius, subject):
    return distance(center, subject) < radius


def distance(feat1, feat2):
    return np.linalg.norm(feat1 - feat2)


def realism(manifold_real, feat_subject):
    feats_real = manifold_real.features
    radii_real = manifold_real.radii
    diff = feats_real - feat_subject
    dists = np.linalg.norm(diff, axis=1)
    eps = 1e-6
    ratios = radii_real / (dists + eps)
    max_realism = float(ratios.max())
    return max_realism


def _compute_pairwise_distances(
    features1: npt.NDArray[np.float32], features2: Optional[npt.NDArray[np.float32]] = None
):
    """
    args:
        X: np.array of shape N x dim
        Y: np.array of shape N x dim
    returns:
        N x N symmetric np.array
    """

    num_X = features1.shape[0]
    if features2 is None:
        num_Y = num_X
    else:
        num_Y = features2.shape[0]
    features1 = features1.astype(np.float32)  # to prevent underflow
    X_norm_square = np.sum(features1**2, axis=1, keepdims=True)

    if features2 is None:
        Y_norm_square = X_norm_square
    else:
        Y_norm_square = np.sum(features2**2, axis=1, keepdims=True)

    X_square = np.repeat(X_norm_square, num_Y, axis=1)
    Y_square = np.repeat(Y_norm_square.T, num_X, axis=0)
    if features2 is None:
        features2 = features1
    XY = np.dot(features1, features2.T)
    diff_square = X_square - 2 * XY + Y_square

    del X_square, Y_square
    # check negative distance
    min_diff_square = diff_square.min()
    if min_diff_square < 0:
        idx = diff_square < 0
        diff_square[idx] = 0
        print(
            "WARNING: %d negative diff_squares found and set to zero, min_diff_square=" % idx.sum(),
            min_diff_square,
        )

    distances = np.sqrt(diff_square)
    del diff_square
    gc.collect()
    return distances
