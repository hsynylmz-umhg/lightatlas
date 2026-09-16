"""End-to-end integration tests for the unified LightAtlas pipeline."""

import pandas as pd

from lightatlas.pipeline import PipelineConfig, run_pipeline


def test_pipeline_end_to_end(tmp_path):
    out_dir = tmp_path / "pipeline_run"
    cfg = PipelineConfig(
        source="synthetic",
        n_curves=40,
        seed=42,
        ae_epochs=2,
        vqvae_epochs=3,
        min_cluster_size=5,
        top_fraction=0.25,
        out_dir=out_dir,
    )

    artifacts = run_pipeline(cfg)

    # 1. Assert artifact existence
    assert artifacts["scores"].exists()
    assert artifacts["gallery"].exists()

    # 2. Verify Parquet columns and row counts
    df = pd.read_parquet(artifacts["scores"])
    assert len(df) == 40
    for col in ["id", "score", "cluster", "rarity"]:
        assert col in df.columns

    # 3. Check <svg elements in HTML gallery (top_fraction 0.25 of 40 = 10 curves)
    html_content = artifacts["gallery"].read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in html_content
    assert html_content.count("<svg") == 10
    assert "http://" not in html_content
    assert "https://" not in html_content


def test_pipeline_deterministic_top1(tmp_path):
    cfg1 = PipelineConfig(
        source="synthetic",
        n_curves=50,
        seed=123,
        ae_epochs=2,
        vqvae_epochs=2,
        out_dir=tmp_path / "run1",
    )
    cfg2 = PipelineConfig(
        source="synthetic",
        n_curves=50,
        seed=123,
        ae_epochs=2,
        vqvae_epochs=2,
        out_dir=tmp_path / "run2",
    )

    art1 = run_pipeline(cfg1)
    art2 = run_pipeline(cfg2)

    df1 = pd.read_parquet(art1["scores"])
    df2 = pd.read_parquet(art2["scores"])

    # Top-1 anomaly ID must be deterministic across independent runs
    assert df1.iloc[0]["id"] == df2.iloc[0]["id"]
