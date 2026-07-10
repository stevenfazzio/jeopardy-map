"""Standalone full-page region treemap with a color-metric dropdown.

Same hierarchy as the report's regions figure (common.dmp_region_tree — the
DataMapPlot topic-tree structure over EVoC layers 5 -> 4 -> 3). The dropdown
recolors the tiles by:

  - Double Jeopardy share  (linear, neutral 0.50)
  - Final Jeopardy lift    (share ÷ corpus FJ rate; log2 color, neutral 1x)
  - Daily Double lift      (observed ÷ expected from round × row; log2, 1x)

Lifts are colored on a log2 scale clipped to [1/4x, 4x] so halving and
doubling are visually symmetric; hover always shows all three linear values.
Output: data/analysis/treemap.html (self-contained).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ANALYSIS_DIR, dmp_region_tree  # noqa: E402

OUT_HTML = ANALYSIS_DIR / "treemap.html"

INK = "#0b0b0b"
SURFACE = "#fcfcfb"
DIV = [[0.0, "#1c5cab"], [0.5, "#f0efec"], [1.0, "#c73635"]]
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def log2_clipped(v: np.ndarray) -> np.ndarray:
    return np.log2(np.clip(np.nan_to_num(v, nan=1.0), 0.25, 4.0))


tree = dmp_region_tree()
LIFT_TICKS = dict(tickvals=[-2, -1, 0, 1, 2], ticktext=["≤¼×", "½×", "1×", "2×", "≥4×"])
METRICS = [
    (
        "Double Jeopardy share",
        tree["dj_share"].to_numpy(),
        dict(
            cmin=0.15,
            cmid=0.5,
            cmax=0.85,
            tickvals=[0.2, 0.35, 0.5, 0.65, 0.8],
            ticktext=["0.20", "0.35", "0.50", "0.65", "0.80"],
            title="DJ share",
        ),
    ),
    (
        "Final Jeopardy lift",
        log2_clipped(tree["fj_lift"].to_numpy()),
        dict(cmin=-2, cmid=0, cmax=2, title="FJ lift", **LIFT_TICKS),
    ),
    (
        "Daily Double lift (placement-adjusted)",
        log2_clipped(tree["dd_lift"].to_numpy()),
        dict(cmin=-2, cmid=0, cmax=2, title="DD lift", **LIFT_TICKS),
    ),
]

first = METRICS[0][2]
fig = go.Figure(
    go.Treemap(
        ids=tree["id"],
        labels=tree["label"],
        parents=tree["parent"],
        values=tree["n"],
        branchvalues="total",
        sort=False,
        maxdepth=3,
        marker=dict(
            colors=METRICS[0][1],
            colorscale=DIV,
            cmin=first["cmin"],
            cmid=first["cmid"],
            cmax=first["cmax"],
            line=dict(color=SURFACE, width=2),
            colorbar=dict(
                title=dict(text=first["title"], font=dict(size=12)),
                tickvals=first["tickvals"],
                ticktext=first["ticktext"],
                thickness=12,
                outlinewidth=0,
            ),
        ),
        textfont=dict(family=FONT, size=14),
        customdata=tree[["n", "dj_share", "dd_lift", "fj_lift"]].to_numpy(),
        hovertemplate="<b>%{label}</b><br>n=%{customdata[0]:,.0f} · DJ share %{customdata[1]:.2f}"
        "<br>DD lift %{customdata[2]:.2f}× · FJ lift %{customdata[3]:.2f}×<extra></extra>",
        tiling=dict(pad=2),
        pathbar=dict(visible=True, thickness=24, textfont=dict(size=13)),
    )
)

buttons = [
    dict(
        label=name,
        method="restyle",
        args=[
            {
                "marker.colors": [vals],
                "marker.cmin": spec["cmin"],
                "marker.cmid": spec["cmid"],
                "marker.cmax": spec["cmax"],
                "marker.colorbar.title.text": spec["title"],
                "marker.colorbar.tickvals": [spec["tickvals"]],
                "marker.colorbar.ticktext": [spec["ticktext"]],
            }
        ],
    )
    for name, vals, spec in METRICS
]

fig.update_layout(
    template=None,
    autosize=True,
    paper_bgcolor=SURFACE,
    font=dict(family=FONT, color=INK, size=13),
    title=dict(
        text="Jeopardy clue regions — EVoC hierarchy, Toponymy names"
        "<br><sup>click a tile to drill in, the breadcrumb to come back; "
        "'Minor subtopics' = a region's unclustered space</sup>",
        font=dict(size=17),
        x=0,
        xanchor="left",
    ),
    margin=dict(l=12, r=12, t=76, b=12),
    updatemenus=[
        dict(
            type="dropdown",
            buttons=buttons,
            active=0,
            x=1.0,
            xanchor="right",
            y=1.06,
            yanchor="bottom",
            bgcolor=SURFACE,
            bordercolor="#c3c2b7",
            font=dict(family=FONT, size=13),
        )
    ],
)

body = fig.to_html(
    full_html=False,
    include_plotlyjs="inline",
    config={"displayModeBar": False, "responsive": True},
    default_height="100%",
    default_width="100%",
)
html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Jeopardy clue region treemap</title>
<style>html, body {{ margin: 0; height: 100%; background: {SURFACE}; }} .fig {{ height: 100vh; }}</style>
</head><body><div class="fig">{body}</div></body></html>"""

tmp = str(OUT_HTML) + ".tmp"
Path(tmp).write_text(html)
Path(tmp).replace(OUT_HTML)
print(f"wrote {OUT_HTML} ({len(html) / 1e6:.1f} MB)")
