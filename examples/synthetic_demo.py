"""Standalone demonstration script executing the end-to-end LightAtlas pipeline."""

import time
from pathlib import Path

from lightatlas.pipeline import PipelineConfig, run_pipeline


def main() -> None:
    print("=== LightAtlas Synthetic Pipeline Demo ===")
    t_start = time.perf_counter()

    cfg = PipelineConfig(
        source="synthetic",
        n_curves=100,
        seed=42,
        ae_epochs=3,
        vqvae_epochs=5,
        min_cluster_size=5,
        top_fraction=0.10,
        out_dir=Path("runs/synthetic_demo"),
    )

    artifacts = run_pipeline(cfg)
    elapsed = time.perf_counter() - t_start

    print(f"Pipeline finished successfully in {elapsed:.2f}s (< 15s)")
    print(f"Scores Parquet: {artifacts['scores']}")
    print(f"Gallery HTML:   {artifacts['gallery']}")


if __name__ == "__main__":
    main()
