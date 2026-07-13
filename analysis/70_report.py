"""Assemble the ambient-space analysis into a single self-contained HTML report.

Reads the persisted artifacts under data/analysis/ (plus clue_rows and the
embeddings/kNN cache for two cheap recomputations: the DD placement grid, the
near-duplicate recycling cadence, and the AMI crosswalk) and writes
data/analysis/report.html — prose + plotly figures + data-table twins.

This is a working document for the author, not a blog post.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    ANALYSIS_DIR,
    DRIFT_PARQUET,
    EVOC_LABELS_PARQUET,
    PROBES_PARQUET,
    SWEEP_PARQUET,
    ambient_knn,
    atomic_write_parquet,
    dmp_region_tree,
    load_clue_ids,
    load_clues,
    load_map_labels,
    top1_neighbor_sim,
)

OUT_HTML = ANALYSIS_DIR / "report.html"

# palette roles (dataviz reference palette, light mode; validated)
BLUE = "#2a78d6"  # honest signal
AQUA = "#1baf7a"  # calibration controls
GRAY = "#898781"  # tautology / mechanical channels
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
SURFACE = "#fcfcfb"
PLANE = "#f9f9f7"
DIV = [[0.0, "#1c5cab"], [0.5, "#f0efec"], [1.0, "#c73635"]]  # blue <-> red, neutral mid
SEQ = [[0.0, "#cde2fb"], [0.25, "#86b6ef"], [0.5, "#3987e5"], [0.75, "#256abf"], [1.0, "#0d366b"]]
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


def vline(fig, x, text=None, row=None, col=None):
    fig.add_vline(x=x, line_color=BASE, line_width=1, row=row, col=col)
    if text:
        fig.add_annotation(
            x=np.log10(x) if fig.layout.xaxis.type == "log" else x,
            y=1.0,
            yref="y domain",
            text=text,
            showarrow=False,
            font=dict(size=11, color=MUTED),
            yanchor="bottom",
        )


def details_table(df: pd.DataFrame, caption: str) -> str:
    return (
        f"<details><summary>data table — {caption}</summary>"
        f"{df.to_html(index=False, float_format=lambda v: f'{v:,.3f}', border=0, classes='dt')}"
        f"</details>"
    )


def fig_html(fig: go.Figure, first: bool = False) -> str:
    return fig.to_html(
        full_html=False,
        include_plotlyjs="inline" if first else False,
        config={"displayModeBar": False, "responsive": True},
    )


# ---------------------------------------------------------------- load data
T0 = time.time()
print("loading artifacts ...", flush=True)
sweep = pd.read_parquet(SWEEP_PARQUET)
robust = pd.read_parquet(ANALYSIS_DIR / "batch_robustness.parquet")
probes = pd.read_parquet(PROBES_PARQUET)
drift = pd.read_parquet(DRIFT_PARQUET)
regions = pd.read_parquet(ANALYSIS_DIR / "evoc_region_stats.parquet").set_index("region")
evoc = pd.read_parquet(EVOC_LABELS_PARQUET)

ids = load_clue_ids()
clues = load_clues(ids, columns=["round", "board_row", "daily_double", "season"])
map_lab = load_map_labels(ids)
nbr = ambient_knn()

sections: list[str] = []
first_fig = True


def add_section(anchor: str, kicker: str, title: str, prose: str, blocks: list[str]) -> None:
    body = "\n".join(blocks)
    sections.append(
        f'<section class="card" id="{anchor}"><div class="kicker">{kicker}</div><h2>{title}</h2>{prose}{body}</section>'
    )


def render(fig: go.Figure) -> str:
    global first_fig
    html = fig_html(fig, first=first_fig)
    first_fig = False
    return html


# ---------------------------------------------------------------- F1: sweep
print("building sweep figures ...", flush=True)
TAUT = {
    ("delivery", "Celebrity"),
    ("delivery", "Clue Crew"),
    ("visual_clue", "raw"),
    ("is_repeat_clue", "raw"),
    ("category_recurrence", "rank assortativity"),
}
cat = sweep[sweep["kind"].isin(["categorical", "binary"])].copy()
cat = cat[~cat["variant"].isin(["Standard", "Regular"])]
cat["label"] = cat["field"] + " · " + cat["variant"]
cat["cls"] = [
    "control" if f == "random_control" else ("flagged" if (f, v) in TAUT else "signal")
    for f, v in zip(cat["field"], cat["variant"])
]
order = {
    "round": 0,
    "daily_double": 1,
    "game_type": 2,
    "season": 3,
    "delivery": 4,
    "visual_clue": 5,
    "is_repeat_clue": 6,
    "random_control": 7,
}
cat = cat.sort_values(["field", "lift"], key=lambda s: s.map(order) if s.name == "field" else s, ascending=[True, True])

f1a = go.Figure()
for cls, color, name in [
    ("signal", BLUE, "geometry signal"),
    ("flagged", GRAY, "tautology / mechanical channel"),
    ("control", AQUA, "calibration control"),
]:
    sub = cat[cat["cls"] == cls]
    f1a.add_trace(
        go.Scatter(
            x=sub["lift"],
            y=sub["label"],
            mode="markers",
            name=name,
            marker=dict(color=color, size=9, line=dict(color=SURFACE, width=2)),
            customdata=np.stack([sub["stat_obs"], sub["null_mean"], sub["z"], sub["n_pos"]], axis=-1),
            hovertemplate="%{y}<br>lift %{x:.2f}x · obs %{customdata[0]:.4f} vs null %{customdata[1]:.4f}"
            "<br>z=%{customdata[2]:.1f} · n=%{customdata[3]:,}<extra></extra>",
        )
    )
f1a.update_xaxes(
    type="log",
    tickvals=[0.5, 1, 2, 5, 10, 20],
    title=dict(text="same-label neighbor rate ÷ permutation null (log)", font=dict(size=12)),
)
layout(f1a, height=30 * len(cat) + 120, title="Label fields: same-label clustering vs shuffle null (k=25)")
vline(f1a, 1.0, "chance")

cont = sweep[sweep["kind"] == "continuous"].copy().sort_values("stat_obs")
cont["cls"] = ["flagged" if (f, "rank assortativity") in TAUT else "signal" for f in cont["field"]]
f1b = go.Figure()
for cls, color, name in [("signal", BLUE, "geometry signal"), ("flagged", GRAY, "tautology / mechanical channel")]:
    sub = cont[cont["cls"] == cls]
    f1b.add_trace(
        go.Scatter(
            x=sub["stat_obs"],
            y=sub["field"],
            mode="markers",
            name=name,
            marker=dict(color=color, size=9, line=dict(color=SURFACE, width=2)),
            customdata=np.stack([sub["z"], sub["n_pos"]], axis=-1),
            hovertemplate="%{y}<br>rank assortativity r = %{x:.3f} · z=%{customdata[0]:.0f}"
            " · n=%{customdata[1]:,}<extra></extra>",
        )
    )
f1b.update_xaxes(
    range=[0, 0.7], title=dict(text="rank assortativity r (point vs mean of 25 neighbors)", font=dict(size=12))
)
layout(f1b, height=30 * len(cont) + 120, title="Continuous fields: neighborhood assortativity")

add_section(
    "sweep",
    "metadata × geometry sweep",
    "What does the embedding space know?",
    """<p>Every (non-tag-derived) metadata field, tested against the exact cosine k=25 nearest-neighbor
    graph of the raw 1024-d Cohere vectors. Label fields: the rate at which a clue's neighbors share its
    label, divided by a label-permutation null (<b>lift</b>; 1.0 = chance). Continuous fields: correlation
    between a clue's value and its neighbors' mean (rank-transformed). The synthetic random control lands
    at lift 1.00 (z=+0.2), so the machinery is calibrated. Gray rows are real measurements whose mechanism
    is partly circular: the embedded text literally contains the delivery/visual parenthetical, repeats are
    near-verbatim duplicates, and recurring categories share their embedded category name.</p>
    <p>Headline honest signals: <b>answer_freq</b> (r=0.62 — stock answers live in specific neighborhoods),
    answer/clue <b>length</b> (r≈0.55–0.59), <b>value</b> (r=0.32), and every game structure field
    (round, daily double, board row). The big categorical lifts (game_type, season) looked spectacular here
    but are episode-batch artifacts — dissected two sections down.</p>""",
    [
        render(f1a),
        details_table(
            cat[["field", "variant", "n_pos", "base_rate", "stat_obs", "null_mean", "z", "lift"]], "label-field sweep"
        ),
        render(f1b),
        details_table(cont[["field", "n_pos", "stat_obs", "z"]], "continuous sweep"),
    ],
)

# ------------------------------------------------- F2: batch robustness dumbbell
print("building robustness figure ...", flush=True)
rob = robust.copy()
rob["field"] = rob["field"].str.replace(" (episode-modal)", "", regex=False)
naive_map = {}
for _, r in sweep[sweep["kind"] == "categorical"].iterrows():
    naive_map[(r["field"], r["variant"])] = r["lift"]
naive_map[("visual_clue", "visual")] = float(sweep.query("field=='visual_clue'")["lift"].iloc[0])

rows = []
for _, r in rob.iterrows():
    if r["field"] == "season" or r["class"] in ("Standard", "not visual", "Regular"):
        continue
    naive = naive_map.get((r["field"], r["class"]))
    if naive is None:
        continue
    rows.append(
        {
            "label": f"{r['field']} · {r['class']}",
            "naive": naive,
            "batch": r["lift_vs_batch_null"],
            "z": r["z"],
            "level": r["null_level"],
        }
    )
dumb = pd.DataFrame(rows).sort_values("naive")

f2 = go.Figure()
for _, r in dumb.iterrows():
    f2.add_trace(
        go.Scatter(
            x=[r["naive"], r["batch"]],
            y=[r["label"], r["label"]],
            mode="lines",
            line=dict(color=GRID, width=2),
            showlegend=False,
            hoverinfo="skip",
        )
    )
f2.add_trace(
    go.Scatter(
        x=dumb["naive"],
        y=dumb["label"],
        mode="markers",
        name="clue-level null (naive)",
        marker=dict(color=GRAY, size=9, line=dict(color=SURFACE, width=2)),
        hovertemplate="%{y}<br>naive lift %{x:.2f}x<extra></extra>",
    )
)
f2.add_trace(
    go.Scatter(
        x=dumb["batch"],
        y=dumb["label"],
        mode="markers",
        name="batch-level null (honest)",
        marker=dict(color=BLUE, size=9, line=dict(color=SURFACE, width=2)),
        customdata=np.stack([dumb["z"], dumb["level"]], axis=-1),
        hovertemplate="%{y}<br>batch-corrected lift %{x:.2f}x (z=%{customdata[0]:.1f}, "
        "null permutes %{customdata[1]}s)<extra></extra>",
    )
)
f2.update_xaxes(
    type="log", tickvals=[0.5, 1, 2, 5, 10, 20], title=dict(text="same-label neighbor lift (log)", font=dict(size=12))
)
layout(f2, height=30 * len(dumb) + 130, title="The episode/category-batch correction collapses most 'signatures'")
vline(f2, 1.0, "chance")

add_section(
    "batch",
    "confound control",
    "Content arrives in batches — and that ate the headline numbers",
    """<p>A Jeopardy category is ~5 clues on one topic inside one episode, so any label that is assigned
    at episode or category granularity clusters trivially at the clue level. The honest null permutes
    labels at the field's natural assignment level: whole episodes trade labels (game_type, season) or
    whole (episode, category) batches trade their internal label patterns (delivery, visual_clue). Same-batch
    edges then keep matching labels under every permutation, moving the mechanical component into the null.</p>
    <p>Result: tournament topical signatures mostly vanish (College Championship 1.14×, Champions Wildcard
    1.09× survive mildly; ToC, Invitational, All-Star come out <i>more generic</i> than random episode sets),
    while Clue Crew (4.2×), visual clues (4.3×), and celebrity-read clues (2.3×) keep genuine cross-batch
    structure — with the parenthetical-tautology caveat. A methods note: my first attempt used size-matched
    pattern swaps for episodes too, which silently pinned primetime formats (120–180-clue episode_ids) onto
    each other and produced artifactual negatives — worth remembering.</p>""",
    [render(f2), details_table(dumb[["label", "naive", "batch", "z", "level"]], "naive vs batch-corrected lifts")],
)

# ---------------------------------------------------------- F3+F4: daily doubles
print("building daily-double figures ...", flush=True)
mr = clues[clues["round"].isin(["jeopardy", "double_jeopardy"])]
grid = (
    mr.assign(row=mr["board_row"].astype(int))
    .groupby(["round", "row"])["daily_double"]
    .mean()
    .unstack("row")
    .reindex(["jeopardy", "double_jeopardy"])
)
f3 = go.Figure(
    go.Heatmap(
        z=grid.to_numpy() * 100,
        x=[f"row {c}" for c in grid.columns],
        y=["Jeopardy!", "Double Jeopardy!"],
        colorscale=SEQ,
        xgap=2,
        ygap=2,
        texttemplate="%{z:.1f}%",
        textfont=dict(size=12),
        colorbar=dict(title=dict(text="DD rate %", font=dict(size=11)), thickness=12, outlinewidth=0),
        hovertemplate="%{y}, %{x}: %{z:.2f}% of clues are DDs<extra></extra>",
    )
)
layout(f3, height=230, title="Where daily doubles sit on the board (the mechanical part)")

solid = regions[(regions["dd_exp"] >= 20) & (regions.index != "Unlabelled")]
dd_show = pd.concat([solid.nlargest(8, "dd_z"), solid.nsmallest(8, "dd_z")]).sort_values("dd_lift")
f4 = go.Figure(
    go.Bar(
        x=dd_show["dd_lift"],
        y=dd_show.index,
        orientation="h",
        marker=dict(
            color=[("#c73635" if v > 1 else "#1c5cab") for v in dd_show["dd_lift"]], line=dict(color=SURFACE, width=2)
        ),
        customdata=np.stack([dd_show["dd_obs"], dd_show["dd_exp"], dd_show["dd_z"], dd_show["n"]], axis=-1),
        hovertemplate="%{y}<br>%{customdata[0]:.0f} DDs vs %{customdata[1]:.1f} expected from board position"
        "<br>lift %{x:.2f}x · z=%{customdata[2]:.1f} · n=%{customdata[3]:,}<extra></extra>",
    )
)
f4.update_xaxes(
    type="log",
    tickvals=[0.2, 0.5, 1, 2],
    title=dict(text="observed DDs ÷ expected from round × row (log)", font=dict(size=12)),
)
layout(
    f4,
    height=30 * len(dd_show) + 120,
    title="Daily-double hot spots and dead zones, net of round × row placement (EVoC regions)",
)
vline(f4, 1.0, "expected")

dd_probe = probes.query("field == 'daily_double (main rounds)'")["value"].iloc[0]
add_section(
    "dd",
    "daily doubles",
    "Writers hide daily doubles in the academic corners",
    f"""<p>Placement is mechanical first — DDs live in rows 3–5, at roughly double the rate in Double
    Jeopardy (top panel). So the topical test compares each region's observed DD count against the sum of
    its members' P(DD | round × row). Net of placement, the geography/monarchy/Shakespeare/planets side of
    the space runs ~1.5–2× hot while TV/sports/rock/food runs 0.16–0.36×. A linear probe on the raw vectors
    predicts DD at <b>AUC {dd_probe:.3f}</b> under episode-grouped CV (shuffled control 0.504) — the clue text
    alone carries real information about where the writers put them.</p>""",
    [
        render(f3),
        render(f4),
        details_table(
            dd_show.reset_index()[["region", "n", "dd_obs", "dd_exp", "dd_lift", "dd_z"]], "DD hot spots / dead zones"
        ),
    ],
)

# ------------------------------------------------------------- F5: curriculum
print("building curriculum figure ...", flush=True)
cur = regions[(regions.index != "Unlabelled") & (regions["n"] >= 550)].copy()
lab_pick = set(
    list(cur.nlargest(4, "dj_share").index)
    + list(cur.nsmallest(4, "dj_share").index)
    + list(cur.nlargest(3, "fj_lift").index)
    + list(cur[cur["fj_lift"] == 0].index)
)
cur["text"] = [r if r in lab_pick else "" for r in cur.index]
cur["tpos"] = [
    "middle left" if x > 0.7 else ("bottom center" if r == "Spirits and Cocktail Ingredients" else "top center")
    for r, x in zip(cur.index, cur["dj_share"])
]
f5 = go.Figure(
    go.Scatter(
        x=cur["dj_share"],
        y=cur["fj_lift"],
        mode="markers+text",
        text=cur["text"],
        textposition=cur["tpos"],
        textfont=dict(size=10.5, color=INK2),
        cliponaxis=False,
        marker=dict(
            color=BLUE,
            size=np.clip(np.sqrt(cur["n"]) / 2.4, 7, 24),
            sizemode="diameter",
            line=dict(color=SURFACE, width=2),
            opacity=0.9,
        ),
        customdata=np.stack([cur.index, cur["n"], cur["dd_lift"]], axis=-1),
        hovertemplate="<b>%{customdata[0]}</b><br>DJ share %{x:.2f} · FJ lift %{y:.2f}x"
        "<br>n=%{customdata[1]:,} · DD lift %{customdata[2]:.2f}x<extra></extra>",
    )
)
f5.update_xaxes(
    title=dict(text="share of main-round clues that are Double Jeopardy (0.50 = neutral)", font=dict(size=12))
)
f5.update_yaxes(
    title=dict(text="Final Jeopardy enrichment (1.0 = neutral)", font=dict(size=12)),
    range=[-0.35, float(cur["fj_lift"].max()) * 1.12],  # room below 0 for the zero-FJ labels
)
layout(f5, height=560, title="The curriculum gradient: rounds sort the space from snacks to sonatas")
vline(f5, 0.5)
f5.add_hline(y=1.0, line_color=BASE, line_width=1)

add_section(
    "rounds",
    "rounds",
    "Round is written into the topic mix",
    """<p>Each dot is an EVoC region (≥550 clues; size ∝ clue count). Right = Double-Jeopardy-skewed,
    up = Final-Jeopardy-enriched. Food, sports, fast food, and gaming sit hard left (DJ share 0.15–0.26)
    and mostly never make Final Jeopardy (culinary and cocktails: literally zero FJs); classical
    composers, modern art, mythology, and European geography sit right and high. Same-round kNN lifts on
    the full corpus: Jeopardy 1.19×, Double Jeopardy 1.10×, and Final Jeopardy <b>3.4×</b> in the ambient
    space vs only 1.7× on the 2D map — FJ has a stylistic signature (long, ornate, single-answer prestige
    clues) that the projection smooths away.</p>""",
    [
        render(f5),
        details_table(
            cur.reset_index()[["region", "n", "dj_share", "dj_z", "fj_lift", "dd_lift"]].sort_values("dj_share"),
            "curriculum scatter",
        ),
    ],
)

# ------------------------------------------------------------------ F6-8: time
print("building time figures ...", flush=True)
cd = drift[drift["analysis"] == "centroid_drift"].copy()
cd["sd"] = (cd["stat"] - cd["null_mean"]) / cd["z"]
f6 = make_subplots(
    rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.06, subplot_titles=["all clues", "Regular play only"]
)
for i, variant in enumerate(["all clues", "Regular play only"], start=1):
    sub = cd[cd["variant"] == variant]
    x = sub["unit"].str.replace("->", "→").tolist()
    hi, lo = sub["null_mean"] + 2 * sub["sd"], (sub["null_mean"] - 2 * sub["sd"]).clip(lower=0)
    f6.add_trace(
        go.Scatter(
            x=x + x[::-1],
            y=pd.concat([hi, lo[::-1]]) * 1e4,
            mode="lines",
            fill="toself",
            fillcolor="rgba(137,135,129,0.18)",
            line=dict(width=0),
            hoverinfo="skip",
            name="shuffle null ±2σ",
            showlegend=(i == 1),
        ),
        row=1,
        col=i,
    )
    f6.add_trace(
        go.Scatter(
            x=x,
            y=sub["stat"] * 1e4,
            mode="lines+markers",
            name="observed",
            showlegend=(i == 1),
            line=dict(color=BLUE, width=2),
            marker=dict(size=8, color=BLUE, line=dict(color=SURFACE, width=2)),
            customdata=sub["z"],
            hovertemplate="%{x}<br>cos distance %{y:.2f} ×10⁻⁴ (z=%{customdata:.0f})<extra></extra>",
        ),
        row=1,
        col=i,
    )
f6.update_yaxes(title=dict(text="centroid cosine distance ×10⁻⁴", font=dict(size=12)), row=1, col=1)
layout(f6, height=400, title="Season-to-season centroid drift: unambiguous, and tiny", hovermode="x unified")
f6.update_layout(margin=dict(t=76), legend=dict(y=1.14))  # keep legend clear of subplot titles

ss = drift[drift["analysis"] == "same_season_knn"].copy()
ss["season"] = ss["unit"]
rob_season = robust[robust["field"] == "season"].set_index("class")["lift_vs_batch_null"]
ss["batch_lift"] = [rob_season.get(u.lstrip("S"), np.nan) for u in ss["unit"]]
f7 = go.Figure()
f7.add_trace(
    go.Bar(
        x=ss["season"],
        y=ss["lift"],
        name="vs clue-level null (naive)",
        marker=dict(color=GRAY, line=dict(color=SURFACE, width=2)),
        hovertemplate="%{x}: naive lift %{y:.2f}x<extra></extra>",
    )
)
f7.add_trace(
    go.Bar(
        x=ss["season"],
        y=ss["batch_lift"],
        name="vs episode-swap null (honest)",
        marker=dict(color=BLUE, line=dict(color=SURFACE, width=2)),
        hovertemplate="%{x}: batch-corrected lift %{y:.2f}x<extra></extra>",
    )
)
f7.add_hline(y=1.0, line_color=BASE, line_width=1)
f7.update_yaxes(title=dict(text="same-season neighbor lift", font=dict(size=12)))
layout(
    f7,
    height=380,
    title="Same-season clustering is episode structure; net of it, seasons anti-cluster",
    barmode="group",
    bargap=0.25,
    barcornerradius=4,
)

seasons = clues["season"].astype(int).to_numpy()
top1 = nbr[:, 0]
sim1 = top1_neighbor_sim()
m = sim1 >= 0.9
gaps = np.abs(seasons[m] - seasons[top1[m]])
obs_gap = np.bincount(gaps, minlength=10) / m.sum()
shares = np.bincount(seasons - seasons.min()) / len(seasons)
exp_gap = np.zeros(10)
for i, si in enumerate(shares):
    for j, sj in enumerate(shares):
        exp_gap[abs(i - j)] += si * sj
f8 = go.Figure()
f8.add_trace(
    go.Bar(
        x=list(range(10)),
        y=obs_gap * 100,
        name="near-duplicate pairs (top-1 nbr, cos ≥ 0.9)",
        marker=dict(color=BLUE, line=dict(color=SURFACE, width=2)),
        hovertemplate="gap %{x} seasons: %{y:.1f}% of near-dupe pairs<extra></extra>",
    )
)
f8.add_trace(
    go.Scatter(
        x=list(range(10)),
        y=exp_gap * 100,
        mode="lines+markers",
        name="random-pair expectation",
        line=dict(color=GRAY, width=2),
        marker=dict(size=8, line=dict(color=SURFACE, width=2)),
        hovertemplate="gap %{x}: %{y:.1f}% expected<extra></extra>",
    )
)
f8.update_xaxes(title=dict(text="seasons between a clue and its near-duplicate", font=dict(size=12)), dtick=1)
f8.update_yaxes(title=dict(text="% of pairs", font=dict(size=12)))
layout(f8, height=380, title="What the 2016+ window shows — right-censored at gap ~9", barcornerradius=4)

# F8b/F8c: the archive truth (45_recycling_archive.py artifacts; lexical = primary).
# The windowed figure above is kept deliberately: its "7–8 season clock" was the
# censored projection of the S40 strike event, and the contrast is the lesson.
recyc = pd.read_parquet(ANALYSIS_DIR / "recycling_archive.parquet")
strike_m = pd.read_parquet(ANALYSIS_DIR / "recycling_strike_months.parquet")
f8b = go.Figure()
f8b.add_vrect(x0=9.5, x1=24.5, fillcolor=GRID, opacity=0.4, line_width=0)
f8b.add_annotation(
    x=17,
    y=1.0,
    yref="y domain",
    text="invisible to the 2016+ window",
    showarrow=False,
    font=dict(size=11, color=MUTED),
    yanchor="top",
)
f8b.add_hline(y=1.0, line_color=BASE, line_width=1)
for scope, lab, color in [
    ("s<32", "S1–31 (1984–2015)", BLUE),
    ("s32-39", "S32–39 (2015–2023)", AQUA),
    ("s40-41", "S40–41 (2023–2025, strike era)", "#eda100"),
]:
    t = recyc.query("method == 'lexical_nearest' and scope == @scope and gap <= 24")
    f8b.add_trace(
        go.Scatter(
            x=t["gap"],
            y=t["ratio"],
            mode="lines+markers",
            name=lab,
            line=dict(color=color, width=2),
            marker=dict(size=6),
            hovertemplate=lab + " — gap %{x}: %{y:.2f}× (n=%{customdata})<extra></extra>",
            customdata=t["n_pairs"],
        )
    )
f8b.update_xaxes(
    title=dict(
        text="seasons back to the most recent earlier source (full archive, lexical near-dupes)", font=dict(size=12)
    ),
    dtick=2,
)
f8b.update_yaxes(title=dict(text="observed ÷ expected", font=dict(size=12)), rangemode="tozero")
layout(f8b, height=400, title="Archive truth: a 1–2 season dead zone, a long tail, and a strike-era reach")

rate_s = recyc.query("method == 'lexical_rate' and scope == 'per_season' and gap >= 1")
f8c = go.Figure(
    go.Bar(
        x=1983 + rate_s["gap"],
        y=rate_s["observed_share"] * 100,
        marker=dict(color=[BLUE if s in (24, 40) else BASE for s in rate_s["gap"]], line=dict(color=SURFACE, width=2)),
        hovertemplate="season starting %{x}: %{y:.1f}% recycled<extra></extra>",
    )
)
f8c.add_annotation(
    x=2007, y=6.4, text="2007–08 WGA strike", showarrow=False, font=dict(size=11, color=INK2), yanchor="bottom"
)
f8c.add_annotation(
    x=2023, y=13.5, text="2023 WGA strike", showarrow=False, font=dict(size=11, color=INK2), yanchor="bottom"
)
f8c.update_yaxes(title=dict(text="% of aired clues", font=dict(size=12)))
layout(f8c, height=340, title="Recycled share by season: both writers' strikes are visible", barcornerradius=4)

air_r2 = probes.query("field == 'air_year'")["value"].iloc[0]
add_section(
    "time",
    "time",
    "A show that avoids its own recent past — until a strike",
    f"""<p>Four views. (1) Every consecutive season pair's centroid moves more than a label-shuffle
    null (z +9 to +26) — but the movement is ~4×10⁻⁴ cosine per season, ~2×10⁻³ over the whole decade:
    real, and tiny. The Regular-only panel shows the S39→S40 bump survives removing tournaments (season 40
    = the strike season). (2) The naive same-season lift (1.2–2.0×) inverts under the episode-swap null:
    net of episode structure, a season's clues are slightly <i>less</i> similar to each other than to the
    rest of the decade — S41 at 0.87× (z=−8.5). (3) The windowed near-dupe cadence (kept below as a
    methods exhibit) once read as a "7–8 season clock" — that claim is <b>retired</b>: gaps are
    right-censored at ~9 in a 10-season window, and the full-archive analysis
    (<code>45_recycling_archive.py</code>, text-only) shows the stationary policy is a 1–2 season dead
    zone (o/e 0.08–0.63) with a broad 3–15 season reach and a long tail — median nearest-source gap 9,
    49% of recycling ≥10 seasons back. (4) The apparent "clock" was mostly the <b>2023 WGA strike</b>:
    S40 recycled 13.3% of aired clues (Nov 2023 peaks at 47%) from two deep back-catalog bands, and the
    <b>2007–08 strike</b> shows the same signature (S24 at 6.3%; Jun 2008 at 20%), each lagging its
    strike by the taping lead. Consistently, a linear probe reads air date at only R²={air_r2:.3f}
    under episode-grouped CV.</p>""",
    [
        render(f6),
        render(f7),
        render(f8c),
        render(f8b),
        render(f8),
        details_table(
            recyc.query(
                "method == 'lexical_nearest' and scope in ('archive', 's<32', 's32-39', 's40-41') and gap <= 24"
            )[["scope", "gap", "n_pairs", "observed_share", "expected_share", "ratio"]],
            "archive recycling cadence by era",
        ),
        details_table(strike_m, "strike-window monthly recycling"),
    ],
)

# ---------------------------------------------------------------- F9: probes
print("building probes figure ...", flush=True)
pb = probes.copy()
pb["cls"] = np.where(
    pb["field"].str.contains("control"),
    "control",
    np.where(pb["field"].isin(["visual_clue", "is_repeat_clue"]), "flagged", "signal"),
)
auc = pb[pb["metric"].str.contains("AUC")].sort_values("value")
r2 = pb[pb["metric"] == "R2"].sort_values("value")
f9 = make_subplots(rows=1, cols=2, horizontal_spacing=0.16, subplot_titles=["classification (AUC)", "regression (R²)"])
for panel, (col, frame) in enumerate([(1, auc), (2, r2)], start=0):
    for cls, color, name in [
        ("signal", BLUE, "honest probe"),
        ("flagged", GRAY, "tautology / mechanical"),
        ("control", AQUA, "shuffled-label control"),
    ]:
        sub = frame[frame["cls"] == cls]
        if sub.empty:
            continue
        f9.add_trace(
            go.Scatter(
                x=sub["value"],
                y=sub["field"],
                mode="markers",
                name=name,
                legendgroup=cls,
                showlegend=(panel == 0),
                marker=dict(color=color, size=9, line=dict(color=SURFACE, width=2)),
                error_x=dict(array=sub["sd"], color=BASE, thickness=1),
                hovertemplate="%{y}: %{x:.3f} ± " + "%{customdata:.3f}<extra></extra>",
                customdata=sub["sd"],
            ),
            row=1,
            col=col,
        )
f9.add_vline(x=0.5, line_color=BASE, line_width=1, row=1, col=1)
f9.add_vline(x=0.0, line_color=BASE, line_width=1, row=1, col=2)
f9.update_xaxes(range=[0.45, 1.02], row=1, col=1)
layout(f9, height=380, title="Linear probes on the raw vectors (3-fold CV, episode-grouped)")

add_section(
    "probes",
    "linear probes",
    "What a linear readout recovers",
    """<p>Logistic/ridge probes with folds grouped by episode, so category batches never straddle the
    train/test split (ungrouped CV flatters every episode-level field). The clue text linearly encodes its
    own length (R²=0.62), its answer's archive-wide popularity (R²=0.40), the round (macro-AUC 0.72), the
    daily double (0.69), and dollar value (R²=0.11) — and essentially cannot see air date (R²=0.03) or
    special-vs-regular play (AUC 0.54). visual_clue at 0.999 is the parenthetical tautology on full
    display, and is_repeat (0.61) mostly reflects that repeats are old-material re-airs.</p>""",
    [render(f9), details_table(pb[["field", "metric", "value", "sd", "n"]], "probe results")],
)

# ---------------------------------------------------- F10+F11: EVoC inventory
print("building region figures ...", flush=True)
evoc_cols = sorted(c for c in evoc.columns if c.startswith("evoc_label_"))
map_cols = sorted(c for c in map_lab.columns if c.startswith("label_layer_"))
inv = pd.DataFrame(
    {
        "layer": range(len(evoc_cols)),
        "evoc_unlab": [(evoc[c] == "Unlabelled").mean() for c in evoc_cols],
        "map_unlab": [(map_lab[c] == "Unlabelled").mean() for c in map_cols],
        "evoc_regions": [evoc[c].nunique() - 1 for c in evoc_cols],
        "map_regions": [map_lab[c].nunique() - 1 for c in map_cols],
    }
)
f10 = go.Figure()
f10.add_trace(
    go.Bar(
        x=inv["layer"],
        y=inv["evoc_unlab"] * 100,
        name="EVoC on 1024-d vectors",
        marker=dict(color=BLUE, line=dict(color=SURFACE, width=2)),
        customdata=inv["evoc_regions"],
        hovertemplate="layer %{x}: %{y:.0f}% unclustered · %{customdata:,} regions<extra></extra>",
    )
)
f10.add_trace(
    go.Bar(
        x=inv["layer"],
        y=inv["map_unlab"] * 100,
        name="map clusterer on 2D UMAP",
        marker=dict(color=AQUA, line=dict(color=SURFACE, width=2)),
        customdata=inv["map_regions"],
        hovertemplate="layer %{x}: %{y:.0f}% unlabelled · %{customdata:,} regions<extra></extra>",
    )
)
f10.update_xaxes(title=dict(text="hierarchy layer (0 = finest)", font=dict(size=12)), dtick=1)
f10.update_yaxes(title=dict(text="% of clues in no named region", font=dict(size=12)))
layout(
    f10,
    height=380,
    title="The honest space is an archipelago; the projection manufactures density",
    barmode="group",
    bargap=0.25,
    barcornerradius=4,
)

# cached: deterministic given the two label sets; delete the parquet to recompute
AMI_PARQUET = ANALYSIS_DIR / "ami_crosswalk.parquet"
if AMI_PARQUET.exists():
    ami_df = pd.read_parquet(AMI_PARQUET)
else:
    print("computing AMI crosswalk (first run only; cached afterwards) ...", flush=True)
    from sklearn.metrics import adjusted_mutual_info_score  # noqa: E402

    ami_rows = []
    for ec in evoc_cols:
        e = evoc[ec].to_numpy()
        best = None
        for mc in map_cols:
            mvec = map_lab[mc].to_numpy()
            both = (e != "Unlabelled") & (mvec != "Unlabelled")
            ami = adjusted_mutual_info_score(e[both], mvec[both])
            if best is None or ami > best[1]:
                best = (mc, ami, both.mean())
        ami_rows.append(
            {
                "EVoC layer": ec,
                "regions": f"{evoc[ec].nunique() - 1:,}",
                "best map layer": best[0],
                "AMI": round(best[1], 3),
                "doubly-labeled coverage": f"{best[2]:.0%}",
            }
        )
    ami_df = pd.DataFrame(ami_rows)
    atomic_write_parquet(ami_df, AMI_PARQUET)
ami_table = ami_df.to_html(index=False, border=0, classes="dt")

# Nested treemap: DataMapPlot topic-tree structure over EVoC layers 5 -> 4 -> 3,
# built by common.dmp_region_tree() (shared with the standalone treemap page).
tree = dmp_region_tree()

f11 = go.Figure(
    go.Treemap(
        ids=tree["id"],
        labels=tree["label"],
        parents=tree["parent"],
        values=tree["n"],
        branchvalues="total",
        sort=False,
        maxdepth=3,
        marker=dict(
            colors=tree["dj_share"],
            colorscale=DIV,
            cmid=0.5,
            cmin=0.15,
            cmax=0.85,
            line=dict(color=SURFACE, width=2),
            colorbar=dict(title=dict(text="DJ share", font=dict(size=11)), thickness=12, outlinewidth=0),
        ),
        textfont=dict(family=FONT, size=13),
        customdata=tree[["n", "dj_share", "dd_lift", "fj_lift"]].to_numpy(),
        hovertemplate="<b>%{label}</b><br>n=%{customdata[0]:,.0f} · DJ share %{customdata[1]:.2f}"
        "<br>DD lift %{customdata[2]:.2f}x · FJ lift %{customdata[3]:.2f}x<extra></extra>",
        tiling=dict(pad=2),
        pathbar=dict(visible=True, thickness=22, textfont=dict(size=12)),
    )
)
layout(
    f11,
    height=680,
    title="Region hierarchy, mega-regions → story layer (DataMapPlot topic-tree structure; click to drill)",
)

top25 = regions.head(25).reset_index()
top25_html = top25[["region", "n", "dd_lift", "dd_z", "dj_share", "fj_lift", "trend_pp_decade"]].to_html(
    index=False, border=0, classes="dt", float_format=lambda v: f"{v:,.2f}"
)

add_section(
    "regions",
    "EVoC ambient regions",
    "The region inventory of the honest space",
    """<p>EVoC clusters the raw 1024-d vectors directly (19 seconds, no projection), producing a 6-layer
    hierarchy (2,970 → 605 → 217 → 74 → 27 → 5 regions) named by Toponymy + Haiku (~$8.40). Two structural
    facts. First, 44–73% of clues sit in no dense region depending on layer, versus ~13–20% Unlabelled on
    the 2D map: UMAP compresses the diffuse ocean into labelable blobs. "Unlabelled" is the density
    clusterer's per-layer noise set — a feature of the space, not a failure. Second, where both methods do
    commit, they largely agree (AMI 0.86–0.93 at mid layers on doubly-labeled rows) — the map's islands are
    real; it's the ocean it invents. The treemap below is built exactly like DataMapPlot's topic tree:
    each region's parent is its median provenance through the coarser layers' point-level labels, with
    unclustered (−1) as a first-class value — so every "Minor subtopics" tile is a real branch holding
    the named finer regions that sit in its parent's unclustered space (the root-level one contains
    classical music, food, brands, games, cocktails, …), bottoming out in residual leaves for clues that
    stay unclustered at every scale shown. Click a tile to drill; the breadcrumb bar comes back. Clue
    placement is honor-then-residual (ramify's rule), so areas are exact clue counts; color is DJ share,
    and parents show their aggregate.</p>"""
    + ami_table,
    [render(f10), render(f11), f"<h3>Story-layer regions, 25 largest</h3>{top25_html}"],
)

# ------------------------------------------------------------------- assemble
print("assembling html ...", flush=True)
n_clues = len(clues)
tldr = """
<ul class="tldr">
<li><b>Daily doubles are topically deliberate.</b> Net of board position: monarchy/Shakespeare/geography
~2× hot; TV/sports/rock/food 0.16–0.36×. Linear probe AUC 0.69 (control 0.504).</li>
<li><b>Rounds form a curriculum.</b> Food/sports/games → Jeopardy; composers/art/mythology → Double
Jeopardy; Final Jeopardy triples down on prestige and has never (in-window) touched culinary.</li>
<li><b>The show is topically stationary but anti-recent — and both WGA strikes show up in the
recycling record.</b> Decade centroid drift ≈ 0.002 cosine; net of episodes, seasons
<i>anti</i>-cluster; air date is nearly unreadable (probe R²=0.03). Full-archive recycling: a 1–2
season dead zone, a long tail (median source gap 9 seasons), and strike-era spikes — Nov 2023 = 47%
of aired clues recycled, Jun 2008 = 20% (the old windowed "7–8 season clock" is retired as a
censoring artifact).</li>
<li><b>Tournament "topical identities" are mostly episode batching.</b> Only College Championship (1.14×)
and Champions Wildcard (1.09×) survive an episode-swap null.</li>
<li><b>The strongest continuous signals:</b> stock-answer frequency (assortativity 0.62, probe R²=0.40)
and clue/answer length (R² up to 0.62).</li>
<li><b>The 1024-d space is an archipelago</b> — EVoC leaves 44–73% of clues unclustered where the 2D map
leaves ~13–20%; where both label, they agree (AMI up to 0.93).</li>
</ul>"""

methods = f"""
<section class="card" id="methods"><div class="kicker">setup</div><h2>Corpus & methods in one paragraph</h2>
<p>{n_clues:,} clues (2016–2025, seasons 32–41), each embedded once from "Category / Clue / Answer" with
Cohere embed-v4.0 (1024-d, clustering mode) — the same vectors behind the map. All tests run in this
raw space, not the 2D layout. Local structure: exact cosine k=25 nearest-neighbor graph; statistics are
same-label neighbor rates or neighbor-mean correlations against permutation nulls (500/300/200 draws;
seed 42), with nulls chosen to match each field's assignment level — clue shuffles, round×row strata for
daily doubles, episode swaps for episode-level fields, category-batch pattern swaps for within-episode
fields. Global structure: logistic/ridge probes, 3-fold CV grouped by episode. Effect sizes (lift, r,
AUC, R²) are the headline; with n=135k, z-scores are decoration. A random-label control calibrates at
lift 1.00. Fields derived from the dataset's nonstandard <code>topic_tags</code> (subject, difficulty)
are excluded throughout.</p></section>"""

caveats = """
<section class="card" id="caveats"><div class="kicker">read with care</div><h2>Caveats</h2>
<ul>
<li><b>Tautology channels.</b> embed_text keeps the "(X of the Clue Crew presents…)" parenthetical, so
delivery/visual signals partly read a literal marker (probe AUC 0.999). category_recurrence sees its own
category name; repeats are near-verbatim duplicates. Gray marks throughout.</li>
<li><b>Region stats describe the clustered subset.</b> The story layer leaves 64% of clues Unlabelled;
compositions are about the dense islands, not the ocean (whose DD lift is ~1.02 — unremarkable).</li>
<li><b>One embedding model, one text recipe.</b> Category text is embedded, so wordplay/gimmick structure
shapes the space by design; a category-free embedding would move some conclusions.</li>
<li><b>Season trends within regions are weak</b> (|t| mostly &lt; 2 on 10 points) — the honest read is
"remarkably stable", not a growth story.</li>
<li><b>FJ n=2,284</b>; its per-region FJ lifts are noisy outside the biggest regions.</li>
</ul></section>"""

appendix = """
<section class="card" id="appendix"><div class="kicker">appendix</div><h2>Artifacts & reproduction</h2>
<p>Everything lives under <code>data/analysis/</code> (gitignored) and regenerates from the numbered
scripts in <code>analysis/</code>: <code>common.py</code> (loaders, kNN cache, permutation machinery),
<code>10_evoc_toponymy.py</code> (EVoC + Haiku naming; the only paid step, ~$8.40),
<code>20_metadata_sweep.py</code>, <code>30_probes.py</code>, <code>40_temporal_drift.py</code>,
<code>50_evoc_regions.py</code>, <code>60_batch_robustness.py</code>, and this report
(<code>70_report.py</code>). Parquet twins of every figure: metadata_sweep, batch_robustness,
probe_results, season_drift, evoc_region_stats, evoc_labels. kNN graph: ambient_knn.npz (exact, k=25).
toponymy 0.5.0's EVoCClusterer wrapper is incompatible with evoc 0.3.1, so stage 10 drives evoc.EVoC
directly and hands Toponymy a pre-fitted duck-typed clusterer.</p></section>"""

css = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body { margin: 0; background: #f9f9f7; color: #0b0b0b;
  font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }
.wrap { max-width: 980px; margin: 0 auto; padding: 40px 20px 80px; }
header.hero { margin: 8px 0 20px; }
header.hero h1 { font-size: 27px; line-height: 1.25; margin: 4px 0 8px; }
header.hero .sub { color: #52514e; max-width: 62ch; }
.kicker { font-size: 11px; letter-spacing: 0.09em; text-transform: uppercase; color: #898781; }
.card { background: #fcfcfb; border: 1px solid rgba(11,11,11,0.10); border-radius: 12px;
  padding: 22px 24px 16px; margin: 18px 0; }
.card h2 { font-size: 19px; margin: 2px 0 10px; }
.card h3 { font-size: 15px; margin: 18px 0 6px; }
.card p, .card li { color: #24231f; max-width: 75ch; }
ul.tldr { padding-left: 20px; } ul.tldr li { margin: 7px 0; }
code { background: #f0efec; padding: 1px 5px; border-radius: 4px; font-size: 13px; }
details { margin: 4px 0 14px; }
details summary { cursor: pointer; color: #52514e; font-size: 13px; }
table.dt { border-collapse: collapse; font-size: 12.5px; margin: 10px 0;
  font-variant-numeric: tabular-nums; display: block; overflow-x: auto; max-height: 420px; }
table.dt th { text-align: left; color: #52514e; font-weight: 600; border-bottom: 1px solid #c3c2b7;
  padding: 5px 12px 5px 0; position: sticky; top: 0; background: #fcfcfb; }
table.dt td { border-bottom: 1px solid #e1e0d9; padding: 4px 12px 4px 0; }
nav.toc { font-size: 13px; color: #52514e; margin-bottom: 4px; }
nav.toc a { color: #2a78d6; text-decoration: none; margin-right: 14px; }
footer { color: #898781; font-size: 12.5px; margin-top: 28px; }
"""

toc = (
    '<nav class="toc"><a href="#methods">setup</a><a href="#sweep">sweep</a><a href="#batch">batch nulls</a>'
    '<a href="#dd">daily doubles</a><a href="#rounds">rounds</a><a href="#time">time</a>'
    '<a href="#probes">probes</a><a href="#regions">regions</a><a href="#caveats">caveats</a>'
    '<a href="#appendix">appendix</a></nav>'
)

html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Jeopardy clue embedding-space analysis — working notes</title>
<style>{css}</style></head>
<body><div class="wrap">
<header class="hero">
<div class="kicker">jeopardy-map · analysis working notes · {time.strftime("%Y-%m-%d")}</div>
<h1>What the clue embedding space knows</h1>
<p class="sub">A tour of the ambient (1024-d) geometry of {n_clues:,} Jeopardy clues, 2016–2025:
what each metadata field looks like from inside the embedding space, which apparent signals are
mechanical, and what the honest region inventory says. Companion to the interactive map; for internal
use, not publication.</p>
{toc}
</header>
<section class="card"><div class="kicker">tl;dr</div><h2>Findings at a glance</h2>{tldr}</section>
{methods}
{"".join(sections)}
{caveats}
{appendix}
<footer>Generated by analysis/70_report.py · seeds fixed at 42 · nulls: label permutation
(500/300/200 draws) at the field's assignment level · plotly, single self-contained file.</footer>
</div></body></html>"""

tmp = str(OUT_HTML) + ".tmp"
Path(tmp).write_text(html)
Path(tmp).replace(OUT_HTML)
print(f"wrote {OUT_HTML} ({len(html) / 1e6:.1f} MB in {time.time() - T0:.1f}s)")
