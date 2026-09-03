"""Predice el cluster de un cliente hipotetico en vivo, sin depender de
un pickle de sklearn: K-Means en inferencia es solo "escalar y buscar
el centroide mas cercano", así que basta con los parametros guardados
por train.py (mean/scale del StandardScaler + los 4 centroides)."""
import numpy as np


def scale_input(values: dict, features: list, mean: list, scale: list) -> np.ndarray:
    raw = np.array([values[f] for f in features], dtype=float)
    return (raw - np.array(mean)) / np.array(scale)


def predict_cluster(values: dict, features: list, mean: list, scale: list, centroids: list):
    """Devuelve (cluster_id, distancias_a_cada_centroide)."""
    x_scaled = scale_input(values, features, mean, scale)
    centroids_arr = np.array(centroids)
    dists = np.linalg.norm(centroids_arr - x_scaled, axis=1)
    cluster_id = int(np.argmin(dists))
    return cluster_id, dists
