"""Render the interactive DataMapPlot of Jeopardy clues.

Layout: one point per clue, positioned by semantic similarity (UMAP of the
embed-v4.0 embeddings of "Category / Clue / Answer"). A colormap dropdown spans
the clue's own fields (air date, round, difficulty, daily double, value, length);
hover shows the full clue; clicking a point runs a web search of the answer;
Toponymy region names (stage 04) float over the clusters if present.

Inputs:  data/umap_coords.npz, data/clue_rows.parquet,
         [optional] data/toponymy_labels.parquet
Output:  data/clue_map.html  (+ copied to docs/index.html)
"""

from __future__ import annotations

from html import escape
from urllib.parse import quote

import datamapplot
import numpy as np
import pandas as pd
from config import (
    CLUE_ROWS_PARQUET,
    DOCS_HTML,
    MAP_HTML,
    PROJECT_NAME,
    PROJECT_TAGLINE,
    TOPONYMY_LABELS_PARQUET,
    UMAP_COORDS_NPZ,
)

ROUND_LABELS = {
    "jeopardy": "Jeopardy",
    "double_jeopardy": "Double Jeopardy",
    "final_jeopardy": "Final Jeopardy",
}
ROUND_COLORS = {
    "Jeopardy": "#1f4e79",
    "Double Jeopardy": "#b45309",
    "Final Jeopardy": "#7a1f1f",
    "Other": "#9aa0a6",
}


def main():
    crd = np.load(UMAP_COORDS_NPZ, allow_pickle=True)
    coords = crd["coords"].astype(np.float32)
    layout = pd.DataFrame({"clue_id": crd["clue_id"], "x": coords[:, 0], "y": coords[:, 1]})

    meta = pd.read_parquet(CLUE_ROWS_PARQUET)
    df = layout.merge(meta, on="clue_id", how="left")  # left-merge preserves coords order
    assert len(df) == len(layout), "join changed row count — clue_id not unique"
    print(f"Map points: {len(df):,}")

    coords_xy = df[["x", "y"]].to_numpy()

    # Optional region labels (finest first, as Toponymy stores them), aligned by clue_id.
    label_layers = []
    if TOPONYMY_LABELS_PARQUET.exists():
        topo = pd.read_parquet(TOPONYMY_LABELS_PARQUET)
        merged = df[["clue_id"]].merge(topo, on="clue_id", how="left")
        layer_cols = sorted(
            (c for c in topo.columns if c.startswith("label_layer_")),
            key=lambda s: int(s.rsplit("_", 1)[1]),
        )
        label_layers = [merged[c].fillna("Unlabelled").to_numpy() for c in layer_cols]
        print(f"  using {len(label_layers)} Toponymy label layer(s)")
    else:
        print("  no Toponymy labels; rendering without region names")

    # --- hover fields ---
    cat = df["category"].fillna("")
    clue = df["clue_text"].fillna("")
    ans = df["answer"].fillna("")
    air = pd.to_datetime(df["air_date"], errors="coerce")
    air_str = air.dt.strftime("%Y-%m-%d").fillna("?")
    round_disp = df["round"].map(ROUND_LABELS).fillna("Other")
    value = df["value"]
    value_str = value.map(lambda v: f"${int(v):,}" if pd.notna(v) and v > 0 else "—")
    diff = df["difficulty"]
    diff_str = diff.map(lambda d: f"{int(d)}/5" if pd.notna(d) else "—")
    search_url = ["https://www.google.com/search?q=" + quote(f"{a} jeopardy") for a in ans]

    extra = pd.DataFrame(
        {
            "category": [escape(c) for c in cat],
            "clue": [escape(c) for c in clue],
            "answer": [escape(a) for a in ans],
            "air": air_str.to_numpy(),
            "round": round_disp.to_numpy(),
            "value": value_str.to_numpy(),
            "difficulty": diff_str.to_numpy(),
            "search_url": search_url,
        }
    )

    hover_template = (
        '<div style="max-width:320px">'
        '<div style="font-weight:700;font-size:11px;letter-spacing:.04em;'
        'text-transform:uppercase;opacity:.65">{category}</div>'
        '<div style="margin-top:3px;font-size:14px;line-height:1.35">{clue}</div>'
        '<div style="margin-top:5px;font-weight:700;color:#1f4e79">{answer}</div>'
        '<div style="margin-top:5px;opacity:.8;font-size:12px">'
        "{round} &nbsp;·&nbsp; {value} &nbsp;·&nbsp; {air} &nbsp;·&nbsp; difficulty {difficulty}</div>"
        "</div>"
    )

    # Search box matches on the answer.
    hover_text = ans.to_numpy()

    # --- colormaps (dropdown) ---
    air_num = (air.dt.year + (air.dt.dayofyear - 1) / 366.0).to_numpy()
    air_num = np.where(np.isfinite(air_num), air_num, np.nanmedian(air_num))

    diff_num = diff.to_numpy(dtype=float)
    diff_num = np.where(np.isfinite(diff_num), diff_num, np.nanmedian(diff_num))

    value_num = value.to_numpy(dtype=float)
    value_num = np.where(np.isfinite(value_num) & (value_num > 0), value_num, np.nan)
    value_num = np.where(np.isfinite(value_num), value_num, np.nanmedian(value_num))

    clue_wc = df["clue_word_count"].to_numpy(dtype=float)
    round_cat = round_disp.to_numpy()
    dd_cat = np.where(df["daily_double"].fillna(False).to_numpy(), "Daily Double", "Regular")

    rawdata = [air_num, round_cat, diff_num, dd_cat, value_num, clue_wc]
    metadata = [
        {"field": "air_date", "description": "Air date (year)", "kind": "continuous", "cmap": "viridis"},
        {"field": "round", "description": "Round", "kind": "categorical", "color_mapping": ROUND_COLORS},
        {"field": "difficulty", "description": "Difficulty (1-5)", "kind": "continuous", "cmap": "plasma"},
        {
            "field": "daily_double",
            "description": "Daily Double?",
            "kind": "categorical",
            "color_mapping": {"Daily Double": "#e4572e", "Regular": "#3a4a5c"},
        },
        {"field": "value", "description": "Clue value ($)", "kind": "continuous", "cmap": "cividis"},
        {"field": "clue_word_count", "description": "Clue length (words)", "kind": "continuous", "cmap": "magma"},
    ]

    print("Rendering DataMapPlot...")
    fig = datamapplot.create_interactive_plot(
        coords_xy,
        *label_layers,
        hover_text=hover_text,
        hover_text_html_template=hover_template,
        extra_point_data=extra,
        on_click="window.open(`{search_url}`, '_blank')",
        colormap_rawdata=rawdata,
        colormap_metadata=metadata,
        title=PROJECT_NAME,
        sub_title=PROJECT_TAGLINE,
        enable_search=True,
        font_family="IBM Plex Sans",
        tooltip_font_family="IBM Plex Sans",
        darkmode=False,
        background_color="#ffffff",
    )
    fig.save(str(MAP_HTML))

    print(f"Saved {MAP_HTML} ({MAP_HTML.stat().st_size / 1e6:.1f} MB)")

    DOCS_HTML.parent.mkdir(exist_ok=True)
    DOCS_HTML.write_bytes(MAP_HTML.read_bytes())
    print(f"Copied to {DOCS_HTML}")


if __name__ == "__main__":
    main()
