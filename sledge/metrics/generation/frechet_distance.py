# flake8: noqa
# TODO: Fix this file and remove the noqa comment.
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List

import networkx as nx
import numpy as np
import numpy.typing as npt
import torch
from scipy import linalg
from tuplan_garage.mapping.metrics.generative.graph_features import (
    geometry_edges_lengths,
    geometry_edges_orientation,
    get_edges_graph,
    get_latents,
    get_route_length,
    urban_planning_reach_and_convenience,
)
from tuplan_garage.mapping.metrics.generative.ipr import run_improved_precision_recall


@dataclass
class GraphFeatures:

    route_length: float

    vae_latents: List[npt.NDArray[np.float32]]
    imagenet_latents: List[npt.NDArray[np.float32]]

    length: List[float]
    orientation: List[float]

    connectivity: List[int]
    density: int
    reach: List[int]
    convenience: List[int]

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> GraphFeatures:
        """Implemented. See interface."""
        return GraphFeatures(
            route_length=data["route_length"],
            vae_latents=data["vae_latents"],
            imagenet_latents=data["imagenet_latents"],
            length=data["length"],
            orientation=data["orientation"],
            connectivity=data["connectivity"],
            density=data["density"],
            reach=data["reach"],
            convenience=data["convenience"],
        )


def collect_graph_feature(
    G: nx.DiGraph,
    vae_encoder: torch.nn.Module,
    imagenet_encoder: torch.nn.Module,
    device: torch.device,
) -> GraphFeatures:

    # Route
    route_length = get_route_length(G)

    G_edges, edges = get_edges_graph(G)

    # Learned
    vae_latents = get_latents(edges, vae_encoder, device, num_channels=12)
    imagenet_latents = get_latents(edges, imagenet_encoder, device, num_channels=3)

    # Geometry:
    length = geometry_edges_lengths(edges)
    orientation = geometry_edges_orientation(edges)

    # Urban planning
    connectivity = [degree for _, degree in G_edges.degree()]
    density = len(connectivity)
    reach, convenience = urban_planning_reach_and_convenience(G_edges)

    return GraphFeatures(
        route_length,
        vae_latents,
        imagenet_latents,
        length,
        orientation,
        connectivity,
        density,
        reach,
        convenience,
    )


def flatten_and_numpy(two_dim_list: List[List[Any]]) -> List[Any]:
    output = []
    for dim_list in two_dim_list:
        output.extend(dim_list)
    return np.array(output)


def run_frechet_distance(
    graph_features: List[GraphFeatures], sampled_graph_features: List[GraphFeatures]
) -> Dict[str, float]:

    results: Dict[str, float] = {}

    # Route Length:
    np.array([feature.route_length for feature in graph_features])
    sampled_route_lengths = np.array([feature.route_length for feature in sampled_graph_features])

    results["route_mu"] = sampled_route_lengths.mean()
    results["route_std"] = sampled_route_lengths.std()

    # Learned:
    vae_latents = np.concatenate([feature.vae_latents[None, ...] for feature in graph_features], axis=0)
    vae_latents_sampled = np.concatenate([feature.vae_latents[None, ...] for feature in sampled_graph_features], axis=0)

    vae_precision, vae_recall = run_improved_precision_recall(vae_latents, vae_latents_sampled)
    results["vae"] = calculate_fid(vae_latents, vae_latents_sampled)
    results["vae_precision"] = vae_precision
    results["vae_recall"] = vae_recall

    imagenet_latents = np.concatenate([feature.imagenet_latents[None, ...] for feature in graph_features], axis=0)
    imagenet_latents_sampled = np.concatenate(
        [feature.imagenet_latents[None, ...] for feature in sampled_graph_features], axis=0
    )

    results["imagenet"] = calculate_fid(imagenet_latents, imagenet_latents_sampled)
    imagenet_precision, imagenet_recall = run_improved_precision_recall(imagenet_latents, imagenet_latents_sampled)
    results["imagenet_precision"] = imagenet_precision
    results["imagenet_recall"] = imagenet_recall

    #  Geometry:
    results["lengths"] = calculate_fid(
        flatten_and_numpy([feature.length for feature in graph_features]),
        flatten_and_numpy([feature.length for feature in sampled_graph_features]),
    )

    results["orientation"] = calculate_fid(
        flatten_and_numpy([feature.orientation for feature in graph_features]),
        flatten_and_numpy([feature.orientation for feature in sampled_graph_features]),
    )

    #  Urban planning:
    results["connectivity"] = calculate_fid(
        flatten_and_numpy([feature.connectivity for feature in graph_features]),
        flatten_and_numpy([feature.connectivity for feature in sampled_graph_features]),
    )

    results["density"] = calculate_fid(
        np.array([feature.density for feature in graph_features]),
        np.array([feature.density for feature in sampled_graph_features]),
    )

    results["reach"] = calculate_fid(
        flatten_and_numpy([feature.reach for feature in graph_features]),
        flatten_and_numpy([feature.reach for feature in sampled_graph_features]),
    )

    results["convenience"] = calculate_fid(
        flatten_and_numpy([feature.convenience for feature in graph_features]),
        flatten_and_numpy([feature.convenience for feature in sampled_graph_features]),
    )
    return results


def calculate_fid(features1, features2):
    assert features1.ndim == features2.ndim

    if features1.ndim == 1:
        return calculate_univariate_fid(features1, features2)
    else:
        return calculate_multivariate_fid(features1, features2)


def calculate_univariate_fid(features1, features2):
    assert features1.ndim == features2.ndim
    assert features1.ndim == 1
    # Calculate mean and covariance for both sets of features
    mu1, var1, std1 = features1.mean(), features1.var(), features1.std()
    mu2, var2, std2 = features2.mean(), features2.var(), features2.std()

    fid_squared = (mu1 - mu2) ** 2 + 2 * (var1 + var2 - 2 * std1 * std2)

    return fid_squared**0.5


def calculate_multivariate_fid(features1, features2):
    assert features1.ndim == features2.ndim
    assert features1.ndim > 1
    # Calculate mean and covariance for both sets of features
    mu1, sigma1 = features1.mean(axis=0), np.cov(features1, rowvar=False)
    mu2, sigma2 = features2.mean(axis=0), np.cov(features2, rowvar=False)

    # Calculate Fréchet distance
    fid = calculate_multivariate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid


def calculate_multivariate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    # Calculate the squared difference in means
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2
    # product might be almost singular
    square = sigma1.dot(sigma2)
    covmean, _ = linalg.sqrtm(square, disp=False)

    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            np.max(np.abs(covmean.imag))
            # raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    fid_squared = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    return fid_squared**0.5
