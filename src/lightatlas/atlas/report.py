"""Zero-network, self-contained HTML and inline SVG gallery generator."""

from pathlib import Path

import numpy as np
import pandas as pd


def _svg_curve(f: np.ndarray, w: int = 360, h: int = 120) -> str:
    """Render an inline, self-contained SVG curve without external network dependencies.

    Scales flux values between 1st and 99th percentiles. Contains zero external URLs.

    Parameters
    ----------
    f : np.ndarray
        Array of flux values.
    w : int, default=360
        SVG view width.
    h : int, default=120
        SVG view height.

    Returns
    -------
    str
        Self-contained inline SVG string.
    """
    f_arr = np.asarray(f, dtype=np.float64)
    n = len(f_arr)
    if n < 2:
        return f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}"></svg>'

    q01, q99 = np.percentile(f_arr, [1.0, 99.0])
    span = q99 - q01
    if span <= 1e-12:
        span = 1.0

    pad_x = 8.0
    pad_y = 10.0
    eff_w = w - 2.0 * pad_x
    eff_h = h - 2.0 * pad_y

    step = max(1, n // 512)
    indices = np.arange(0, n, step)

    points = [
        f"{pad_x + (i / (n - 1)) * eff_w:.1f},"
        f"{h - pad_y - np.clip((f_arr[i] - q01) / span, 0.0, 1.0) * eff_h:.1f}"
        for i in indices
    ]
    points_str = " ".join(points)

    svg = (
        f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}">'
        f'<rect width="{w}" height="{h}" rx="6" fill="#121620" />'
        f'<polyline fill="none" stroke="#00e5ff" stroke-width="1.5" points="{points_str}" />'
        f"</svg>"
    )
    return svg


def render_gallery(rows: pd.DataFrame, out_path: Path | str) -> Path:
    """Render a standalone, beautifully styled dark-theme HTML gallery.

    Embeds inline SVGs for each light curve. Guaranteed strictly zero external
    network calls, zero web fonts, and zero CDN links.

    Parameters
    ----------
    rows : pd.DataFrame
        DataFrame with columns 'id', 'score', 'cluster', and 'f' (flux arrays).
    out_path : Path or str
        Destination HTML filepath.

    Returns
    -------
    Path
        Resolved destination path.

    Raises
    ------
    ValueError
        If required columns are missing from the DataFrame.
    """
    required = {"id", "score", "cluster", "f"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"rows DataFrame is missing required columns: {missing}")

    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    cards_html = []
    for _, row in rows.iterrows():
        cid = str(row["id"])
        score_val = float(row["score"])
        cluster_val = int(row["cluster"])
        flux_arr = np.asarray(row["f"], dtype=np.float64)

        cluster_tag = f"Cluster {cluster_val}" if cluster_val >= 0 else "Outlier (-1)"
        badge_cls = "badge-outlier" if cluster_val < 0 else "badge-cluster"
        svg_markup = _svg_curve(flux_arr)

        card = f"""
        <div class="card">
            <div class="card-header">
                <span class="card-id">{cid}</span>
                <span class="badge {badge_cls}">{cluster_tag}</span>
            </div>
            <div class="card-svg">
                {svg_markup}
            </div>
            <div class="card-footer">
                <span class="score-label">Anomaly Score:</span>
                <span class="score-value">{score_val:.4f}</span>
            </div>
        </div>
        """
        cards_html.append(card)

    cards_joined = "\n".join(cards_html)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LightAtlas: Anomaly Gallery</title>
    <style>
        :root {{
            --bg-color: #0a0c10;
            --card-bg: #151922;
            --border-color: #232936;
            --text-main: #e2e8f0;
            --text-muted: #8892b0;
            --accent-cyan: #00e5ff;
            --accent-rose: #ff3366;
            --badge-bg: #1e2533;
        }}
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: system-ui, -apple-system, sans-serif;
        }}
        body {{
            background-color: var(--bg-color);
            color: var(--text-main);
            padding: 2rem;
        }}
        header {{
            margin-bottom: 2rem;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 1rem;
        }}
        h1 {{
            font-size: 1.8rem;
            color: var(--accent-cyan);
            margin-bottom: 0.5rem;
        }}
        p.subtitle {{
            color: var(--text-muted);
            font-size: 0.95rem;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
            gap: 1.5rem;
        }}
        .card {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1rem;
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        }}
        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .card-id {{
            font-weight: 600;
            color: var(--text-main);
            font-size: 0.95rem;
        }}
        .badge {{
            font-size: 0.75rem;
            padding: 0.2rem 0.6rem;
            border-radius: 4px;
            font-weight: 500;
        }}
        .badge-cluster {{
            background-color: var(--badge-bg);
            color: var(--accent-cyan);
        }}
        .badge-outlier {{
            background-color: rgba(255, 51, 102, 0.15);
            color: var(--accent-rose);
        }}
        .card-svg {{
            display: flex;
            justify-content: center;
            align-items: center;
            background: #121620;
            border-radius: 6px;
            overflow: hidden;
        }}
        svg {{
            display: block;
            width: 100%;
            height: auto;
        }}
        .card-footer {{
            display: flex;
            justify-content: space-between;
            font-size: 0.85rem;
            color: var(--text-muted);
            margin-top: auto;
        }}
        .score-value {{
            font-weight: 600;
            color: var(--accent-cyan);
        }}
    </style>
</head>
<body>
    <header>
        <h1>LightAtlas Anomaly Gallery</h1>
        <p class="subtitle">Offline, zero-network interactive light curve morphology report</p>
    </header>
    <main class="grid">
        {cards_joined}
    </main>
</body>
</html>
"""

    # Strictly assert zero external network URLs exist in the output
    assert "http://" not in html_content, "Output HTML contains unexpected http:// link"
    assert "https://" not in html_content, "Output HTML contains unexpected https:// link"

    destination.write_text(html_content, encoding="utf-8")
    return destination
