"""Conservative graph construction and entity-resolution clustering boundary (Phase 8)."""

from shopee_match.clustering.graph import (
    ClusterAssignment,
    GraphDiagnostics,
    ScoredPair,
    build_conservative_clusters,
    score_candidate_pairs,
)
from shopee_match.clustering.metrics import clustering_metrics, edge_metrics

__all__ = [
    "ClusterAssignment",
    "GraphDiagnostics",
    "ScoredPair",
    "build_conservative_clusters",
    "clustering_metrics",
    "edge_metrics",
    "score_candidate_pairs",
]
