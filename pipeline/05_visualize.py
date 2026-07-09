"""Render the interactive DataMapPlot of Jeopardy clues.

Layout: one point per clue, positioned by semantic similarity (UMAP of the
embed-v4.0 embeddings of "Category / Clue / Answer"), in dark mode (near-black
background). A colormap dropdown spans the clue's own fields (air date, round,
difficulty, daily double, value, length, subject, game/tournament, answer &
category frequency, board row); hovering shows the clue as a Jeopardy-blue clue
card (uppercase white clue text, gold answer) plus subject/tournament/delivery
context; clicking a point runs a web search of the answer; Toponymy region names
(stage 04) float over the clusters.

Inputs:  data/umap_coords.npz, data/clue_rows.parquet,
         [optional] data/toponymy_labels.parquet
Output:  data/clue_map.html  (+ copied to docs/index.html)
"""

from __future__ import annotations

from html import escape
from urllib.parse import quote

import datamapplot
import glasbey
import matplotlib as mpl
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
from matplotlib.colors import LinearSegmentedColormap, to_hex

ROUND_LABELS = {
    "jeopardy": "Jeopardy",
    "double_jeopardy": "Double Jeopardy",
    "final_jeopardy": "Final Jeopardy",
}
# Categorical colors are tuned for the dark background: bright enough to read on
# near-black, with "default"/majority values in receding greys.
ROUND_COLORS = {
    "Jeopardy": "#5b9bd5",
    "Double Jeopardy": "#e08214",
    "Final Jeopardy": "#d1495b",
    "Other": "#9aa0a6",
}

JEOPARDY_BLUE = "#060ce9"
GOLD = "#ffd166"

# The hover tooltip styled as a Jeopardy clue card.
CLUE_CARD_CSS = f"""
            font-size: 0.8em;
            font-family: IBM Plex Sans;
            font-weight: 300;
            color: #ffffff !important;
            background-color: {JEOPARDY_BLUE}f0 !important;
            border: 2px solid #04066e;
            border-radius: 6px;
            backdrop-filter: blur(6px);
            box-shadow: 3px 4px 14px #000000aa;
            max-width: 25%;
"""


def categorical_color_mapping(values, default=None, default_color="#585d66"):
    """Build a {value: hex} mapping for a categorical field using a glasbey palette.
    The dominant `default` value (e.g. 'Untagged'/'Regular') is pinned to a muted grey
    so the meaningful categories stand out instead of one color swamping the map.
    The lightness floor keeps every palette color legible on the dark background."""
    uniques = sorted(set(map(str, values)))
    others = [v for v in uniques if v != default]
    palette = glasbey.create_palette(palette_size=max(len(others), 1), lightness_bounds=(40, 90))
    mapping = {v: to_hex(palette[i]) for i, v in enumerate(others)}
    if default is not None and default in uniques:
        mapping[default] = default_color
    return mapping


def _fill_nonfinite(a):
    """Replace NaN/inf with the median so DataMapPlot's continuous scale stays valid."""
    a = np.asarray(a, dtype=float)
    return np.where(np.isfinite(a), a, np.nanmedian(a))


def _clip_p99(a):
    """Winsorize the top tail. The length/value/frequency fields are right-skewed:
    without this, a few extreme outliers stretch the color scale and crush the bulk
    of the points into its bottom fifth (p10-p90 of answer length spans 5% - 20% of
    the raw range). The legend max becomes the p99 value, with the tail pinned there."""
    return np.minimum(a, np.nanpercentile(a, 99))


def truncated_cmap(name, lo, hi=1.0):
    """Register and return a copy of a matplotlib colormap with its low end cut off.
    Tiny points colored from a cmap's near-black end are invisible on the dark map
    background (as near-white ends are on a white one), so every sequential scale
    here is truncated (and reversed where needed) to run dim-but-visible -> bright;
    datamapplot resolves cmap names through the matplotlib registry, so a
    registered truncation is usable by name."""
    base = mpl.colormaps[name]
    trunc = LinearSegmentedColormap.from_list(f"{name}_trunc", base(np.linspace(lo, hi, 256)))
    mpl.colormaps.register(trunc, force=True)
    return trunc.name


def _details_html(subject, game_type, clue_order, repeat_count):
    bits = []
    if subject and subject != "Untagged":
        bits.append(escape(subject.replace("_", " ").title()))
    if game_type and game_type != "Regular":
        bits.append(escape(game_type))
    if pd.notna(clue_order):
        bits.append(f"clue&nbsp;{int(clue_order)}")
    if repeat_count and int(repeat_count) > 0:
        bits.append(f"seen&nbsp;{int(repeat_count) + 1}×")
    if not bits:
        return ""
    return '<div style="margin-top:5px;font-size:11px;opacity:.7">' + " &nbsp;·&nbsp; ".join(bits) + "</div>"


def _delivery_html(delivery, presenter, visual):
    if delivery == "Standard":
        return ""
    who = escape(presenter) if presenter else ""
    if delivery == "Clue Crew":
        label = "Clue Crew" + (f" — {who}" if who else "")
        if visual:
            label += " · on-screen / location"
    elif delivery == "Celebrity":
        label = f"Presented by {who}" if who else "Celebrity-presented"
    else:
        label = "Special delivery"
    return f'<div style="margin-top:5px;font-size:11px;color:{GOLD}">{label}</div>'


def _aside_html(aside):
    if not aside:
        return ""
    a = escape(aside if len(aside) <= 180 else aside[:177] + "…")
    return f'<div style="margin-top:5px;font-size:11px;font-style:italic;opacity:.6">“{a}”</div>'


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

    # New hover lines (pre-formatted in pandas so empty fields render nothing).
    details_html = [
        _details_html(s, g, o, r)
        for s, g, o, r in zip(df["subject"], df["game_type"], df["clue_order"], df["repeat_count"])
    ]
    delivery_html = [_delivery_html(d, p, v) for d, p, v in zip(df["delivery"], df["presenter"], df["visual_clue"])]
    aside_html = [_aside_html(a) for a in df["host_aside"].fillna("")]

    extra = pd.DataFrame(
        {
            "category": [escape(c) for c in cat],
            "clue": [escape(c) for c in clue],
            "answer": [escape(a) for a in ans],
            "air": air_str.to_numpy(),
            "round": round_disp.to_numpy(),
            "value": value_str.to_numpy(),
            "difficulty": diff_str.to_numpy(),
            "details": details_html,
            "deliv": delivery_html,
            "aside": aside_html,
            "search_url": search_url,
        }
    )

    # Styled as a Jeopardy clue card (with CLUE_CARD_CSS): uppercase white clue
    # text on the blue card, answer in gold.
    hover_template = (
        '<div style="max-width:320px">'
        '<div style="font-weight:700;font-size:11px;letter-spacing:.04em;'
        'text-transform:uppercase;opacity:.85">{category}</div>'
        '<div style="margin-top:6px;font-size:13px;line-height:1.45;text-transform:uppercase;'
        'letter-spacing:.04em;font-weight:600;text-shadow:1px 1px 2px #00000088">{clue}</div>'
        '<div style="margin-top:5px;font-weight:700;color:' + GOLD + '">{answer}</div>'
        '<div style="margin-top:5px;opacity:.8;font-size:12px">'
        "{round} &nbsp;·&nbsp; {value} &nbsp;·&nbsp; {air} &nbsp;·&nbsp; difficulty {difficulty}</div>"
        "{details}{deliv}{aside}"
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
    value_num = _clip_p99(np.where(np.isfinite(value_num), value_num, np.nanmedian(value_num)))

    clue_wc = _clip_p99(df["clue_len_words"].to_numpy(dtype=float))
    round_cat = round_disp.to_numpy()
    dd_cat = np.where(df["daily_double"].fillna(False).to_numpy(), "Daily Double", "Regular")

    # New colormaps. subject/game_type are categorical (glasbey palette, default greyed);
    # the two frequency fields are heavy-tailed so they go on a log scale.
    subject_cat = df["subject"].fillna("Untagged").to_numpy()
    game_cat = df["game_type"].fillna("Regular").to_numpy()
    subject_cmap = categorical_color_mapping(subject_cat, default="Untagged")
    game_cmap = categorical_color_mapping(game_cat, default="Regular")
    answer_freq_log = _clip_p99(np.log10(df["answer_freq"].to_numpy(dtype=float).clip(min=1.0)))
    cat_recur_log = _clip_p99(np.log10(df["category_recurrence"].to_numpy(dtype=float).clip(min=1.0)))
    board_row_num = _fill_nonfinite(df["board_row"].to_numpy(dtype=float))
    ans_len_num = _clip_p99(_fill_nonfinite(df["answer_len_chars"].to_numpy(dtype=float)))

    rawdata = [
        air_num,
        round_cat,
        diff_num,
        dd_cat,
        value_num,
        clue_wc,
        ans_len_num,
        subject_cat,
        game_cat,
        answer_freq_log,
        cat_recur_log,
        board_row_num,
    ]
    metadata = [
        {
            "field": "air_date",
            "description": "Air date (year)",
            "kind": "continuous",
            "cmap": truncated_cmap("viridis", 0.2),
        },
        {"field": "round", "description": "Round", "kind": "categorical", "color_mapping": ROUND_COLORS},
        {
            "field": "difficulty",
            "description": "Difficulty (1-5)",
            "kind": "continuous",
            "cmap": truncated_cmap("plasma", 0.15),
        },
        {
            "field": "daily_double",
            "description": "Daily Double?",
            "kind": "categorical",
            "color_mapping": {"Daily Double": "#e4572e", "Regular": "#5a6a7d"},
        },
        {
            "field": "value",
            "description": "Clue value ($)",
            "kind": "continuous",
            "cmap": truncated_cmap("cividis", 0.2),
        },
        {
            "field": "clue_len_words",
            "description": "Clue length (words)",
            "kind": "continuous",
            "cmap": truncated_cmap("magma", 0.25),
        },
        {
            "field": "answer_len_chars",
            "description": "Answer length (chars)",
            "kind": "continuous",
            "cmap": truncated_cmap("GnBu_r", 0.15),
        },
        {
            "field": "subject",
            "description": "Subject (topic tag)",
            "kind": "categorical",
            "color_mapping": subject_cmap,
        },
        {"field": "game_type", "description": "Game / tournament", "kind": "categorical", "color_mapping": game_cmap},
        {
            "field": "answer_freq",
            "description": "Answer frequency (log, archive)",
            "kind": "continuous",
            "cmap": truncated_cmap("inferno", 0.25),
        },
        {
            "field": "category_recurrence",
            "description": "Category recurrence (log, archive)",
            "kind": "continuous",
            "cmap": truncated_cmap("YlGnBu_r", 0.15),
        },
        {
            "field": "board_row",
            "description": "Board row (1-5)",
            "kind": "continuous",
            "cmap": truncated_cmap("YlOrRd_r", 0.15),
        },
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
        tooltip_css=CLUE_CARD_CSS,
        darkmode=True,
    )
    fig.save(str(MAP_HTML))

    print(f"Saved {MAP_HTML} ({MAP_HTML.stat().st_size / 1e6:.1f} MB)")

    DOCS_HTML.parent.mkdir(exist_ok=True)
    DOCS_HTML.write_bytes(MAP_HTML.read_bytes())
    print(f"Copied to {DOCS_HTML}")


if __name__ == "__main__":
    main()
