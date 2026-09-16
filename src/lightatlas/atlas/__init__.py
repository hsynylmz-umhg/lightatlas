"""Atlas clustering, rarity ranking, and offline HTML gallery generation."""

from lightatlas.atlas.cluster import cluster_latent
from lightatlas.atlas.rank import rarity_score
from lightatlas.atlas.report import render_gallery

__all__ = ["cluster_latent", "rarity_score", "render_gallery"]
