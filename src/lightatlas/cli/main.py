"""Typer command line interface for LightAtlas."""

from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from lightatlas.atlas.cluster import cluster_latent
from lightatlas.atlas.rank import rarity_score
from lightatlas.atlas.report import render_gallery
from lightatlas.core.features import feature_matrix
from lightatlas.models.isolation import IsolationScorer
from lightatlas.synth import random_mixture

app = typer.Typer(help="lightatlas: anomaly atlas for astronomical light curves")


@app.command("synthesize")
def synthesize(
    n: Annotated[int, typer.Option(help="Number of curves to generate")] = 100,
    seed: Annotated[int, typer.Option(help="Random seed for reproducibility")] = 42,
    out: Annotated[Path, typer.Option(help="Output parquet file path")] = Path("synthetic.parquet"),
) -> None:
    """Generate synthetic light curves and store IDs, labels, and fluxes in Parquet format."""
    synth_set = random_mixture(n=n, seed=seed)
    df = pd.DataFrame(
        {
            "id": synth_set.ids,
            "label": synth_set.labels,
            "flux": list(synth_set.f),
        }
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    typer.echo(f"Successfully generated {n} light curves -> {out}")


@app.command("demo")
def demo(
    n: Annotated[int, typer.Option(help="Number of curves for demo run")] = 200,
    seed: Annotated[int, typer.Option(help="Random seed")] = 42,
    out_dir: Annotated[Path, typer.Option(help="Output directory for artifacts")] = Path(
        "runs/demo"
    ),
) -> None:
    """Run end-to-end offline pipeline: synthesize, extract, score, cluster, and build gallery."""
    typer.echo(f"Running LightAtlas demo pipeline with {n} curves...")
    synth_set = random_mixture(n=n, seed=seed)

    typer.echo("Extracting 40 astrophysical features...")
    X = feature_matrix(synth_set.t, synth_set.f)

    typer.echo("Fitting baseline IsolationScorer...")
    scorer = IsolationScorer(contamination=0.05, random_state=seed)
    scorer.fit(X)
    scores = scorer.score(X)

    typer.echo("Clustering morphology and computing rarity...")
    clusters = cluster_latent(X, min_cluster_size=10, random_state=seed)
    rarities = rarity_score(clusters)
    sample_rarities = [rarities.get(int(c), 0.0) for c in clusters]

    df = pd.DataFrame(
        {
            "id": synth_set.ids,
            "label": synth_set.labels,
            "score": scores,
            "cluster": clusters,
            "rarity": sample_rarities,
            "f": list(synth_set.f),
        }
    )
    df = df.sort_values(by="score", ascending=False).reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "scores.parquet"
    df.to_parquet(parquet_path)
    typer.echo(f"Saved scores to {parquet_path}")

    gallery_path = out_dir / "gallery.html"
    render_gallery(df, gallery_path)
    typer.echo(f"Generated zero-network HTML gallery at {gallery_path}")


if __name__ == "__main__":
    app()
