"""Blog-grade standalone figures for the jeopardy-map post.

Reads only cached analysis artifacts (no embeddings, regenerates in ~1s) and
writes each figure as a self-contained HTML file to data/analysis/blog/
(plotly.js from CDN so the files stay small and iframe-embeddable), plus an
index.html preview page that stacks them for local review.

Figures (numbers must match data/claim_inventory.md — that doc is the
fact-check sheet):
  fig1_strike_timeline   share of aired clues that are recycled, by season,
                         with monthly zooms on the two WGA-strike windows
  fig2_cadence_eras      reuse-gap observed/expected by era (dead zone, the
                         strike era's deep reach, the censoring wash)
  fig3_dd_lifts          DD topical lifts net of board position (top/bottom
                         regions, diverging around 1x)
  fig4_round_gradient    region DJ-share and FJ-rate extremes (the curriculum)
  fig5_tournament_debunk naive clue-null lift vs episode-swap lift dumbbells

Style: the dataviz reference palette, light mode, same constants as
70_report.py (BLUE = honest signal, GRAY = flattered/mechanical; diverging
blue<->red around a neutral 1x). Palette choices validated with the skill's
validator (aqua/yellow carry direct labels per the contrast relief rule).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ANALYSIS_DIR  # noqa: E402

BLOG_DIR = ANALYSIS_DIR / "blog"
BLOG_DIR.mkdir(exist_ok=True)

# palette roles (dataviz reference palette, light mode; validated — see 70_report.py)
BLUE = "#2a78d6"
AQUA = "#1baf7a"
YELLOW = "#eda100"
GRAY = "#898781"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
SURFACE = "#fcfcfb"
POLE_HI = "#1c5cab"  # diverging pole, above the neutral midpoint
POLE_LO = "#c73635"  # diverging pole, below
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def layout(fig: go.Figure, height: int, title: str | None = None, **kw) -> go.Figure:
    fig.update_layout(
        height=height,
        template=None,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, color=INK, size=13),
        title=(dict(text=title, font=dict(size=15, color=INK), x=0, xanchor="left") if title else None),
        margin=dict(l=10, r=20, t=48 if title else 16, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0, font=dict(size=12)),
        hoverlabel=dict(font=dict(family=FONT, size=12)),
        **kw,
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=BASE, zeroline=False, tickfont=dict(color=INK2), automargin=True)
    fig.update_yaxes(gridcolor=GRID, linecolor=BASE, zeroline=False, tickfont=dict(color=INK2), automargin=True)
    return fig


def write(fig: go.Figure, name: str, written: list[tuple[str, int]]) -> None:
    path = BLOG_DIR / f"{name}.html"
    fig.write_html(path, include_plotlyjs="cdn", full_html=True, config={"displayModeBar": False, "responsive": True})
    written.append((name, fig.layout.height))
    print(f"  wrote {path.name}")


T0 = time.time()
cadence = pd.read_parquet(ANALYSIS_DIR / "recycling_archive.parquet")
strikes = pd.read_parquet(ANALYSIS_DIR / "recycling_strike_months.parquet")
regions = pd.read_parquet(ANALYSIS_DIR / "evoc_region_stats.parquet")
sweep = pd.read_parquet(ANALYSIS_DIR / "metadata_sweep.parquet")
robust = pd.read_parquet(ANALYSIS_DIR / "batch_robustness.parquet")
written: list[tuple[str, int]] = []

# ------------------------------------------------- F1: strike timeline (lead)
print("fig1: strike timeline ...", flush=True)
rate = cadence.query("method == 'lexical_rate' and scope == 'per_season' and gap >= 1").copy()
rate["year"] = 1983 + rate["gap"]  # season n premieres in fall 1983+n
rate["pct"] = rate["observed_share"] * 100
strike_seasons = {24, 40}
rate["hl"] = rate["gap"].isin(strike_seasons)

f1 = make_subplots(
    rows=2,
    cols=2,
    specs=[[{"colspan": 2}, None], [{}, {}]],
    row_heights=[0.56, 0.44],
    vertical_spacing=0.16,
    horizontal_spacing=0.08,
    subplot_titles=(
        "share of aired clues recycling older material, by season",
        "monthly, around the 2007–08 strike",
        "monthly, around the 2023 strike",
    ),
)
f1.add_trace(
    go.Bar(
        x=rate["year"],
        y=rate["pct"],
        marker=dict(color=[BLUE if h else BASE for h in rate["hl"]], line=dict(color=SURFACE, width=2)),
        hovertemplate="season starting %{x}: %{y:.1f}% recycled<extra></extra>",
        showlegend=False,
    ),
    row=1,
    col=1,
)
f1.add_annotation(
    x=2007,
    y=6.3,
    text="2007–08 WGA strike",
    font=dict(size=12, color=INK2),
    showarrow=False,
    yanchor="bottom",
    row=1,
    col=1,
)
f1.add_annotation(
    x=2023,
    y=13.9,
    text="2023 WGA strike",
    font=dict(size=12, color=INK2),
    showarrow=False,
    yanchor="bottom",
    row=1,
    col=1,
)

for col, (lo, hi, s_start, s_end) in enumerate(
    [("2007-01", "2009-06", "2007-11-05", "2008-02-12"), ("2022-07", "2025-07", "2023-05-02", "2023-09-27")], start=1
):
    m = strikes[(strikes["air_month"] >= lo) & (strikes["air_month"] <= hi)].copy()
    # ISO strings, not Timestamps: plotly parses date strings fine and kaleido's
    # orjson encoder rejects pandas Timestamp objects
    m["x"] = pd.PeriodIndex(m["air_month"], freq="M").to_timestamp().strftime("%Y-%m-%d")
    m["pct"] = m["n_repeats"] / m["n_clues"] * 100
    f1.add_vrect(x0=s_start, x1=s_end, fillcolor=GRID, opacity=0.55, line_width=0, row=2, col=col)
    f1.add_annotation(
        x=(pd.Timestamp(s_start) + (pd.Timestamp(s_end) - pd.Timestamp(s_start)) / 2).strftime("%Y-%m-%d"),
        y=26,  # mid-height in data coords: inside the band the line stays low, peaks sit outside it
        text="writers<br>on strike",
        showarrow=False,
        font=dict(size=11, color=MUTED),
        row=2,
        col=col,
    )
    f1.add_trace(
        go.Scatter(
            x=m["x"],
            y=m["pct"],
            mode="lines",
            line=dict(color=BLUE, width=2),
            hovertemplate="%{x|%b %Y}: %{y:.0f}% recycled<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=col,
    )
    peak = m.loc[m["pct"].idxmax()]
    f1.add_annotation(
        x=peak["x"],
        y=peak["pct"],
        text=f"{peak['pct']:.0f}%",
        showarrow=False,
        yshift=12,
        font=dict(size=12, color=INK),
        row=2,
        col=col,
    )
f1.update_yaxes(title=dict(text="% of aired clues", font=dict(size=12)), row=1, col=1)
f1.update_yaxes(range=[0, 52], row=2, col=1)
f1.update_yaxes(range=[0, 52], row=2, col=2)
f1.update_annotations(font=dict(size=12, color=INK2))
layout(f1, 640, "Reruns on the board: recycled clues spike during both writers' strikes")
write(f1, "fig1_strike_timeline", written)

# ------------------------------------------------- F2: cadence by era
print("fig2: cadence eras ...", flush=True)
ERAS = [
    ("s<32", "1984–2015 (S1–31)", BLUE),
    ("s32-39", "2015–2023 (S32–39)", AQUA),
    ("s40-41", "2023–2025 (S40–41, strike era)", YELLOW),
]
f2 = go.Figure()
f2.add_vrect(x0=9.5, x1=24.5, fillcolor=GRID, opacity=0.4, line_width=0)
f2.add_annotation(
    x=17,
    y=1.0,
    yref="y domain",
    text="invisible to a 10-season window",
    showarrow=False,
    font=dict(size=11, color=MUTED),
    yanchor="top",
)
f2.add_hline(y=1.0, line_color=BASE, line_width=1)
f2.add_annotation(
    x=24.3,
    y=1.0,
    text="time-blind expectation",
    showarrow=False,
    font=dict(size=11, color=MUTED),
    yanchor="bottom",
    xanchor="right",
)
for scope, label, color in ERAS:
    t = cadence.query("method == 'lexical_nearest' and scope == @scope and gap <= 24")
    f2.add_trace(
        go.Scatter(
            x=t["gap"],
            y=t["ratio"],
            mode="lines+markers",
            name=label,
            line=dict(color=color, width=2),
            marker=dict(size=6),
            hovertemplate=label + " — gap %{x} seasons: %{y:.2f}× expected (n=%{customdata})<extra></extra>",
            customdata=t["n_pairs"],
        )
    )
f2.add_annotation(
    x=2.6,
    y=0.12,
    text="← dead zone: fresh material is off-limits",
    showarrow=False,
    font=dict(size=11, color=INK2),
    xanchor="left",
    yanchor="bottom",
)
f2.update_xaxes(
    title=dict(text="seasons between a clue and its most recent earlier source", font=dict(size=12)), dtick=2
)
f2.update_yaxes(title=dict(text="observed ÷ expected", font=dict(size=12)), rangemode="tozero")
layout(f2, 440, "How far back the writers reach: reuse gaps vs a time-blind null")
write(f2, "fig2_cadence_eras", written)

# ------------------------------------------------- F3: DD lifts
print("fig3: DD lifts ...", flush=True)
big = regions[regions["n"] >= 500]
dd = pd.concat([big.nlargest(8, "dd_lift"), big.nsmallest(8, "dd_lift").iloc[::-1]]).sort_values("dd_lift")
f3 = go.Figure()
f3.add_vline(x=1.0, line_color=BASE, line_width=1)
for _, r in dd.iterrows():
    f3.add_shape(type="line", x0=1.0, x1=r["dd_lift"], y0=r["region"], y1=r["region"], line=dict(color=GRID, width=2))
f3.add_trace(
    go.Scatter(
        x=dd["dd_lift"],
        y=dd["region"],
        mode="markers",
        marker=dict(
            size=10, color=[POLE_HI if v >= 1 else POLE_LO for v in dd["dd_lift"]], line=dict(color=SURFACE, width=2)
        ),
        hovertemplate="%{y}: %{x:.2f}× the position-expected DD rate (n=%{customdata:,})<extra></extra>",
        customdata=dd["n"],
        showlegend=False,
    )
)
f3.add_annotation(
    x=0.0,  # log axis: annotation x is log10(value); log10(1) = 0
    y=1.02,
    yref="y domain",
    text="expected from board position alone",
    showarrow=False,
    font=dict(size=11, color=MUTED),
    yanchor="bottom",
)
f3.update_xaxes(
    type="log",
    tickvals=[0.125, 0.25, 0.5, 1, 2],
    ticktext=["⅛×", "¼×", "½×", "1×", "2×"],
    title=dict(text="Daily Double rate vs position-adjusted expectation (log scale)", font=dict(size=12)),
)
layout(f3, 500, "Where Daily Doubles live — after adjusting for the famous row effect")
write(f3, "fig3_dd_lifts", written)

# ------------------------------------------------- F4: round gradient
print("fig4: round gradient ...", flush=True)
dj = pd.concat([big.nlargest(8, "dj_share"), big.nsmallest(8, "dj_share").iloc[::-1]]).sort_values("dj_share")
fj = pd.concat([big.nlargest(8, "fj_lift"), big.nsmallest(8, "fj_lift").iloc[::-1]]).sort_values("fj_lift")
f4 = make_subplots(
    rows=2,
    cols=1,
    vertical_spacing=0.11,
    subplot_titles=("share of a region's main-round clues in Double Jeopardy", "Final Jeopardy rate vs league average"),
)
# overall DJ share of main-round clues, from the sweep's class counts
rnd = sweep.query("field == 'round'").set_index("variant")["n_pos"]
mid_dj = float(rnd["double_jeopardy"] / (rnd["double_jeopardy"] + rnd["jeopardy"]))
for row, (t, xcol, mid) in enumerate([(dj, "dj_share", mid_dj), (fj, "fj_lift", 1.0)], start=1):
    f4.add_vline(x=mid, line_color=BASE, line_width=1, row=row, col=1)
    for _, r in t.iterrows():
        f4.add_shape(
            type="line",
            x0=mid,
            x1=r[xcol],
            y0=r["region"],
            y1=r["region"],
            line=dict(color=GRID, width=2),
            row=row,
            col=1,
        )
    f4.add_trace(
        go.Scatter(
            x=t[xcol],
            y=t["region"],
            mode="markers",
            marker=dict(
                size=10, color=[POLE_HI if v >= mid else POLE_LO for v in t[xcol]], line=dict(color=SURFACE, width=2)
            ),
            hovertemplate="%{y}: %{x:.2f} (n=%{customdata:,})<extra></extra>",
            customdata=t["n"],
            showlegend=False,
        ),
        row=row,
        col=1,
    )
f4.add_annotation(
    x=0.06,
    y="Culinary Terms and Food",
    text="← zero Final Jeopardy clues, 2016–2025",
    showarrow=False,
    font=dict(size=11, color=INK2),
    xanchor="left",
    row=2,
    col=1,
)
f4.update_xaxes(tickformat=".0%", row=1, col=1)
f4.update_xaxes(tickvals=[0, 1, 2, 3], ticktext=["0×", "1×", "2×", "3×"], row=2, col=1)
f4.update_annotations(font=dict(size=12, color=INK2))
layout(f4, 880, "The round curriculum: what Jeopardy schedules where")
write(f4, "fig4_round_gradient", written)

# ------------------------------------------------- F5: tournament debunk
print("fig5: tournament debunk ...", flush=True)
naive = sweep.query("field == 'game_type' and variant != 'Regular'")[["variant", "lift"]].rename(
    columns={"lift": "naive"}
)
honest = robust.query("field == 'game_type (episode-modal)' and `class` != 'Regular'")[["class", "lift_vs_batch_null"]]
merged = naive.merge(honest, left_on="variant", right_on="class").sort_values("naive")
f5 = go.Figure()
f5.add_vline(x=1.0, line_color=BASE, line_width=1)
for _, r in merged.iterrows():
    f5.add_shape(
        type="line",
        x0=r["naive"],
        x1=r["lift_vs_batch_null"],
        y0=r["variant"],
        y1=r["variant"],
        line=dict(color=GRID, width=2),
    )
f5.add_trace(
    go.Scatter(
        x=merged["naive"],
        y=merged["variant"],
        mode="markers",
        name="naive clue-shuffle null (flattered)",
        marker=dict(size=10, color=GRAY, line=dict(color=SURFACE, width=2)),
        hovertemplate="%{y}: %{x:.1f}× under the naive null<extra></extra>",
    )
)
f5.add_trace(
    go.Scatter(
        x=merged["lift_vs_batch_null"],
        y=merged["variant"],
        mode="markers",
        name="episode-swap null (honest)",
        marker=dict(size=10, color=BLUE, line=dict(color=SURFACE, width=2)),
        hovertemplate="%{y}: %{x:.2f}× under the episode-swap null<extra></extra>",
    )
)
f5.update_xaxes(
    type="log",
    tickvals=[0.5, 1, 2, 4, 8, 16],
    ticktext=["½×", "1×", "2×", "4×", "8×", "16×"],
    title=dict(text="same-tournament neighbor lift (log scale)", font=dict(size=12)),
)
layout(f5, 480, "Tournament “topical footprints” are episode structure, not content")
write(f5, "fig5_tournament_debunk", written)

# ------------------------------------------------- preview index
frames = "\n".join(
    f'<h2>{name}</h2><iframe src="{name}.html" style="width:100%;height:{h + 30}px;border:0"></iframe>'
    for name, h in written
)
(BLOG_DIR / "index.html").write_text(
    f"""<!doctype html><html><head><meta charset="utf-8"><title>blog figures preview</title>
<style>body{{font-family:{FONT};background:#f9f9f7;margin:2rem auto;max-width:1000px;color:#0b0b0b}}
h2{{font-size:14px;color:#52514e;margin:2rem 0 .3rem}}</style></head><body>
<h1>jeopardy-map blog figures — preview</h1>
{frames}
</body></html>"""
)
print(f"\nwrote {BLOG_DIR / 'index.html'} (+{len(written)} figures) in {time.time() - T0:.1f}s")
