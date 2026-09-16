"""Tests for latent clustering, rarity ranking, offline gallery, and CLI."""

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from lightatlas.atlas.cluster import cluster_latent
from lightatlas.atlas.rank import rarity_score
from lightatlas.atlas.report import render_gallery
from lightatlas.cli.main import app


def test_cluster_latent_splits_mixtures():
    rng = np.random.default_rng(42)
    # Two well-separated Gaussian clusters
    blob1 = rng.normal(loc=-10.0, scale=0.5, size=(50, 10))
    blob2 = rng.normal(loc=10.0, scale=0.5, size=(50, 10))
    Z = np.vstack([blob1, blob2])

    labels = cluster_latent(Z, min_cluster_size=10, random_state=42)
    assert len(labels) == 100

    unique_clusters = set(labels) - {-1}
    assert len(unique_clusters) >= 2


def test_rarity_increases_with_rareness():
    # Large cluster 0 (100), medium cluster 1 (25), small cluster 2 (4), outliers -1 (2)
    labels = np.array([0] * 100 + [1] * 25 + [2] * 4 + [-1] * 2)
    rarity = rarity_score(labels)

    # Strictly increasing rarity
    assert rarity[0] < rarity[1] < rarity[2] < rarity[-1]


def test_gallery_offline(tmp_path):
    rng = np.random.default_rng(42)
    rows = pd.DataFrame(
        {
            "id": ["c0", "c1", "c2", "c3"],
            "score": [0.12, 0.45, 0.81, 0.98],
            "cluster": [0, 1, 0, -1],
            "f": [rng.normal(1.0, 0.01, size=256) for _ in range(4)],
        }
    )
    html_file = tmp_path / "gallery.html"
    out = render_gallery(rows, html_file)

    assert out.exists()
    content = html_file.read_text(encoding="utf-8")

    assert content.startswith("<!DOCTYPE html>")
    assert content.count("<svg") == 4
    assert "http://" not in content
    assert "https://" not in content


def test_cli_synthesize(tmp_path):
    runner = CliRunner()
    out_file = tmp_path / "synthetic.parquet"
    result = runner.invoke(
        app,
        ["synthesize", "--n", "20", "--seed", "42", "--out", str(out_file)],
    )

    assert result.exit_code == 0
    assert out_file.exists()

    df = pd.read_parquet(out_file)
    assert len(df) == 20
    assert "id" in df.columns
    assert "label" in df.columns
    assert "flux" in df.columns


def test_cli_demo(tmp_path):
    runner = CliRunner()
    out_dir = tmp_path / "demo_test"
    result = runner.invoke(
        app,
        ["demo", "--n", "30", "--seed", "42", "--out-dir", str(out_dir)],
    )

    assert result.exit_code == 0
    assert (out_dir / "scores.parquet").exists()
    assert (out_dir / "gallery.html").exists()

    html_content = (out_dir / "gallery.html").read_text(encoding="utf-8")
    assert "LightAtlas Anomaly Gallery" in html_content
    assert "http://" not in html_content
    assert "https://" not in html_content
