"""Plotly chart builders with one consistent WeshFahmna AI visual system.

Palette notes (validated for CVD separation against the dark chart surface):
- Sentiment is a polarity job: blue (positive) / gray midpoint (neutral) / orange (negative).
- Before/after comparisons use violet (baseline) -> cyan (newer / adjusted).
- Single-series magnitude uses cyan.
"""
from __future__ import annotations

import html
import textwrap

import pandas as pd
import plotly.graph_objects as go

from utils.data_loader import SENTIMENT_ORDER, accent_label, domain_label, format_timestamp

SURFACE = "#0f1729"
TEXT = "#e6ecf5"
MUTED = "#93a1b8"
GRID = "rgba(148, 163, 184, 0.14)"
CYAN = "#0f9fbf"
VIOLET = "#8577ee"
ALERT = "#e0703f"
SENTIMENT_COLORS = {"positive": "#3f8fe0", "neutral": "#6b7488", "negative": "#e0703f"}
FONT = "Inter, 'Segoe UI', system-ui, -apple-system, sans-serif"

CHART_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}


def _style(fig: go.Figure, height: int = 340, legend: bool = False) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=TEXT, size=13),
        margin=dict(l=8, r=16, t=16, b=8),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(color=MUTED, size=12), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#16213a", bordercolor="#2b3a57",
                        font=dict(family=FONT, color=TEXT, size=12), align="left"),
        bargap=0.32,
        bargroupgap=0.08,
    )
    axis = dict(automargin=True, gridcolor=GRID, zerolinecolor="rgba(148,163,184,0.35)", linecolor=GRID,
                tickfont=dict(color=MUTED, size=12), title_font=dict(color=MUTED, size=12))
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    # Explicit margins: automargin alone is not reliable inside Streamlit's iframe-less embed.
    y_labels = [str(v) for tr in fig.data if tr.y is not None for v in tr.y if isinstance(v, str)]
    left = (max(map(len, y_labels)) * 7.4 if y_labels else 34) + (26 if fig.layout.yaxis.title.text else 8)
    bottom = 56 if fig.layout.xaxis.title.text else 34
    if fig.layout.xaxis.tickangle:
        bottom += 26
    fig.update_layout(margin=dict(l=int(left) + 8, r=24, t=44 if legend else 16, b=bottom))
    return fig


def _wrap(text: str, width: int = 60, max_lines: int = 6) -> str:
    """Escape and wrap free text for a Plotly hover label."""
    lines = textwrap.wrap(str(text), width=width) or [""]
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] += " …"
    return "<br>".join(html.escape(line) for line in lines)


def _bar_marker(color):
    return dict(color=color, line=dict(color=SURFACE, width=2), cornerradius=4)


def empty_figure(message: str, height: int = 260) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=html.escape(message), showarrow=False, font=dict(color=MUTED, size=14),
                       xref="paper", yref="paper", x=0.5, y=0.5)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _style(fig, height)


# --------------------------------------------------------------------------- call-level


def call_sentiment_distribution(calls: pd.DataFrame, height: int = 260) -> go.Figure:
    counts = calls["overall_sentiment"].value_counts().reindex(SENTIMENT_ORDER, fill_value=0)
    total = max(int(counts.sum()), 1)
    fig = go.Figure(go.Bar(
        y=[s.title() for s in counts.index], x=counts.values, orientation="h",
        marker=_bar_marker([SENTIMENT_COLORS[s] for s in counts.index]),
        text=[f"{v} ({v / total:.0%})" for v in counts.values], textposition="outside",
        textfont=dict(color=TEXT), cliponaxis=False,
        hovertemplate="%{y}: %{x} call(s)<extra>Call-level overall label</extra>",
    ))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(title="Calls", dtick=1, rangemode="tozero", range=[0, max(counts.max(), 1) * 1.3])
    return _style(fig, height)


def category_mix(calls: pd.DataFrame, column: str, height: int = 300) -> go.Figure:
    label = domain_label if column == "domain" else accent_label
    counts = calls[column].value_counts().sort_values(ascending=True)
    minutes = calls.groupby(column)["duration_min"].sum().reindex(counts.index)
    fig = go.Figure(go.Bar(
        y=[label(v) for v in counts.index], x=counts.values, orientation="h",
        marker=_bar_marker(CYAN), text=counts.values, textposition="outside",
        textfont=dict(color=TEXT), cliponaxis=False, customdata=minutes.round(1).values,
        hovertemplate="%{y}<br>%{x} call(s) · %{customdata} recorded min<extra></extra>",
    ))
    fig.update_xaxes(title="Calls", dtick=1, rangemode="tozero", range=[0, max(counts.max(), 1) * 1.2])
    fig.update_yaxes(showgrid=False)
    return _style(fig, height)


def minutes_by_domain(calls: pd.DataFrame, height: int = 300) -> go.Figure:
    agg = calls.groupby("domain").agg(minutes=("duration_min", "sum"), calls=("call_id", "count"))
    agg = agg.sort_values("minutes")
    fig = go.Figure(go.Bar(
        y=[domain_label(v) for v in agg.index], x=agg["minutes"], orientation="h",
        marker=_bar_marker(CYAN), text=[f"{m:.1f}" for m in agg["minutes"]], textposition="outside",
        textfont=dict(color=TEXT), cliponaxis=False, customdata=agg["calls"],
        hovertemplate="%{y}<br>%{x:.1f} recorded min across %{customdata} call(s)<extra></extra>",
    ))
    fig.update_xaxes(title="Recorded minutes", rangemode="tozero", range=[0, max(agg["minutes"].max(), 1) * 1.2])
    fig.update_yaxes(showgrid=False)
    return _style(fig, height)


def transition_heatmap(calls: pd.DataFrame, height: int = 300) -> go.Figure:
    order = SENTIMENT_ORDER
    matrix = pd.crosstab(calls["opening_sentiment"], calls["closing_sentiment"])
    matrix = matrix.reindex(index=order, columns=order, fill_value=0)
    fig = go.Figure(go.Heatmap(
        z=matrix.values, x=[c.title() for c in order], y=[r.title() for r in order],
        colorscale=[[0, "#131d33"], [1, CYAN]], zmin=0, showscale=False, xgap=3, ygap=3,
        text=matrix.values, texttemplate="%{text}", textfont=dict(color=TEXT, size=16),
        hovertemplate="Opening %{y} → closing %{x}<br>%{z} call(s)<extra></extra>",
    ))
    fig.update_xaxes(title="Closing sentiment (category)", side="bottom", showgrid=False)
    fig.update_yaxes(title="Opening sentiment", autorange="reversed", showgrid=False)
    return _style(fig, height)


def opening_vs_closing(calls: pd.DataFrame, height: int = 300) -> go.Figure:
    opening = calls["opening_sentiment"].value_counts().reindex(SENTIMENT_ORDER, fill_value=0)
    closing = calls["closing_sentiment"].value_counts().reindex(SENTIMENT_ORDER, fill_value=0)
    x = [s.title() for s in SENTIMENT_ORDER]
    fig = go.Figure([
        go.Bar(name="Opening", x=x, y=opening.values, marker=_bar_marker(VIOLET),
               text=opening.values, textposition="outside", textfont=dict(color=TEXT), cliponaxis=False,
               hovertemplate="Opening %{x}: %{y} call(s)<extra></extra>"),
        go.Bar(name="Closing", x=x, y=closing.values, marker=_bar_marker(CYAN),
               text=closing.values, textposition="outside", textfont=dict(color=TEXT), cliponaxis=False,
               hovertemplate="Closing %{x}: %{y} call(s)<extra></extra>"),
    ])
    top = max(opening.max(), closing.max(), 1)
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Calls", dtick=max(1, int(top // 4)), range=[0, top * 1.25])
    fig.update_xaxes(showgrid=False)
    return _style(fig, height, legend=True)


def sentiment_change_by_call(calls: pd.DataFrame, height: int = 360) -> go.Figure:
    df = calls[["call_id", "sentiment_change", "sentiment_transition"]].dropna(subset=["sentiment_change"])
    df = df.sort_values("sentiment_change")
    colors = [SENTIMENT_COLORS["positive"] if v > 0 else SENTIMENT_COLORS["negative"] if v < 0
              else SENTIMENT_COLORS["neutral"] for v in df["sentiment_change"]]
    fig = go.Figure(go.Bar(
        y=df["call_id"], x=df["sentiment_change"], orientation="h", marker=_bar_marker(colors),
        text=[f"{v:+.2f}" for v in df["sentiment_change"]], textposition="outside",
        textfont=dict(color=TEXT), cliponaxis=False,
        customdata=[html.escape(t) for t in df["sentiment_transition"]],
        hovertemplate="%{y}<br>Numeric change: %{x:+.3f}<br>Category: %{customdata}<extra></extra>",
    ))
    span = max(df["sentiment_change"].abs().max() if len(df) else 1, 0.1) * 1.35
    fig.update_xaxes(title="Source-provided sentiment_change (numeric score, not %)", range=[-span, span],
                     zeroline=True, zerolinewidth=1.5)
    fig.update_yaxes(showgrid=False)
    return _style(fig, height)


# --------------------------------------------------------------------------- turn-level


def turn_label_share(sent: pd.DataFrame, height: int = 260) -> go.Figure:
    counts = sent["sentiment_label"].value_counts().reindex(SENTIMENT_ORDER, fill_value=0)
    total = max(int(counts.sum()), 1)
    fig = go.Figure(go.Bar(
        y=[s.title() for s in counts.index], x=counts.values / total * 100, orientation="h",
        marker=_bar_marker([SENTIMENT_COLORS[s] for s in counts.index]),
        text=[f"{v / total:.0%} · {v} turns" for v in counts.values], textposition="outside",
        textfont=dict(color=TEXT), cliponaxis=False, customdata=counts.values,
        hovertemplate="%{y}: %{customdata} of " + str(total)
                      + " scored customer turns (%{x:.1f}%)<extra>Highest model probability</extra>",
    ))
    fig.update_xaxes(title="Share of scored customer turns (%)", range=[0, 118], ticksuffix="%")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return _style(fig, height)


def polarity_strip(sent: pd.DataFrame, height: int = 380) -> go.Figure:
    fig = go.Figure()
    for label in SENTIMENT_ORDER:
        part = sent[sent["sentiment_label"] == label]
        if part.empty:
            continue
        fig.add_trace(go.Scatter(
            x=part["sentiment_polarity"], y=part["call_id"], mode="markers", name=label.title(),
            marker=dict(color=SENTIMENT_COLORS[label], size=9, opacity=0.85, line=dict(color=SURFACE, width=1.5)),
            customdata=list(zip([format_timestamp(s) for s in part["start"]],
                                [_wrap(t, 55, 4) for t in part["text"]],
                                part["sentiment_class_confidence"])),
            hovertemplate="%{y} @ %{customdata[0]}<br>Polarity %{x:+.2f} · " + label
                          + " (p=%{customdata[2]:.2f})<br><br>%{customdata[1]}<extra></extra>",
        ))
    fig.update_xaxes(title="Turn polarity (P(positive) − P(negative))", range=[-1.05, 1.05],
                     zeroline=True, zerolinewidth=1.5)
    fig.update_yaxes(categoryorder="category descending", showgrid=False)
    return _style(fig, height, legend=True)


def call_sentiment_timeline(sent_call: pd.DataFrame, height: int = 340) -> go.Figure:
    df = sent_call.sort_values("start")
    fig = go.Figure()
    fig.add_hrect(y0=0, y1=1.05, fillcolor=SENTIMENT_COLORS["positive"], opacity=0.05, line_width=0)
    fig.add_hrect(y0=-1.05, y1=0, fillcolor=SENTIMENT_COLORS["negative"], opacity=0.05, line_width=0)
    fig.add_trace(go.Scatter(
        x=df["minute_in_call"], y=df["sentiment_polarity"], mode="lines",
        line=dict(color="rgba(147,161,184,0.55)", width=2, shape="spline", smoothing=0.4),
        hoverinfo="skip", showlegend=False,
    ))
    for label in SENTIMENT_ORDER:
        part = df[df["sentiment_label"] == label]
        if part.empty:
            continue
        fig.add_trace(go.Scatter(
            x=part["minute_in_call"], y=part["sentiment_polarity"], mode="markers", name=label.title(),
            marker=dict(color=SENTIMENT_COLORS[label], size=11, line=dict(color=SURFACE, width=2)),
            customdata=list(zip([format_timestamp(s) for s in part["start"]],
                                part["positive"], part["neutral"], part["negative"],
                                [_wrap(t) for t in part["text"]])),
            hovertemplate=("<b>%{customdata[0]}</b> · " + label + " · polarity %{y:+.2f}<br>"
                           "P(pos) %{customdata[1]:.2f} · P(neu) %{customdata[2]:.2f} · "
                           "P(neg) %{customdata[3]:.2f}<br><br>%{customdata[4]}<extra></extra>"),
        ))
    fig.update_xaxes(title="Elapsed time in call (minutes)", rangemode="tozero")
    fig.update_yaxes(title="Polarity", range=[-1.05, 1.05], zeroline=True, zerolinewidth=1.5,
                     tickvals=[-1, -0.5, 0, 0.5, 1])
    return _style(fig, height, legend=True)


# --------------------------------------------------------------------------- model quality


def wer_by_call(calls: pd.DataFrame, height: int = 360) -> go.Figure:
    df = calls.sort_values("call_id")
    fig = go.Figure([
        go.Bar(name="Original WER", x=df["call_id"], y=df["asr_wer_pct"], marker=_bar_marker(VIOLET),
               hovertemplate="%{x}<br>Original WER %{y:.2f}%<extra></extra>"),
        go.Bar(name="Filler-adjusted WER", x=df["call_id"], y=df["asr_adjusted_wer_pct"],
               marker=_bar_marker(CYAN),
               hovertemplate="%{x}<br>Filler-adjusted WER %{y:.2f}%<extra></extra>"),
    ])
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Word error rate (%) — lower is better", ticksuffix="%", rangemode="tozero")
    fig.update_xaxes(showgrid=False, tickangle=-35)
    return _style(fig, height, legend=True)


def der_by_call(calls: pd.DataFrame, threshold: float = 40.0, height: int = 360) -> go.Figure:
    df = calls.sort_values("call_id")
    colors = [ALERT if v >= threshold else CYAN for v in df["diarization_der_pct"]]
    fig = go.Figure(go.Bar(
        x=df["call_id"], y=df["diarization_der_pct"], marker=_bar_marker(colors),
        text=[f"{v:.1f}%" for v in df["diarization_der_pct"]], textposition="outside",
        textfont=dict(color=TEXT, size=11), cliponaxis=False,
        customdata=[accent_label(a) for a in df["accent"]],
        hovertemplate="%{x} · %{customdata}<br>DER %{y:.2f}%<extra></extra>",
    ))
    fig.add_hline(y=threshold, line=dict(color=ALERT, width=1, dash="dot"),
                  annotation_text=f"{threshold:.0f}% caution line", annotation_position="top left",
                  annotation_font=dict(color=MUTED, size=11))
    fig.update_yaxes(title="Diarization error rate (%) — lower is better", ticksuffix="%",
                     range=[0, max(df["diarization_der_pct"].max(), threshold) * 1.18])
    fig.update_xaxes(showgrid=False, tickangle=-35)
    return _style(fig, height)


def der_by_accent(calls: pd.DataFrame, height: int = 260) -> go.Figure:
    agg = calls.groupby("accent").agg(der=("diarization_der_pct", "mean"), n=("call_id", "count"))
    agg = agg.sort_values("der")
    fig = go.Figure(go.Bar(
        y=[accent_label(a) for a in agg.index], x=agg["der"], orientation="h", marker=_bar_marker(CYAN),
        text=[f"{v:.1f}%" for v in agg["der"]], textposition="outside", textfont=dict(color=TEXT),
        cliponaxis=False, customdata=agg["n"],
        hovertemplate="%{y}<br>Mean DER %{x:.2f}% across %{customdata} call(s)<extra></extra>",
    ))
    fig.update_xaxes(title="Mean DER (%) — lower is better", ticksuffix="%",
                     range=[0, max(agg["der"].max(), 1) * 1.25])
    fig.update_yaxes(showgrid=False)
    return _style(fig, height)


def qwen_by_call(quality: pd.DataFrame, height: int = 360) -> go.Figure:
    df = quality.sort_values("call_id")
    fig = go.Figure([
        go.Bar(name="Qwen2.5-3B", x=df["call_id"], y=df["q25_overall"], marker=_bar_marker(VIOLET),
               hovertemplate="%{x}<br>Qwen2.5-3B: %{y:.2f} / 5<extra></extra>"),
        go.Bar(name="Qwen3-4B", x=df["call_id"], y=df["q3_overall"], marker=_bar_marker(CYAN),
               hovertemplate="%{x}<br>Qwen3-4B: %{y:.2f} / 5<extra></extra>"),
    ])
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Overall rubric score (out of 5)", range=[0, 5.2], dtick=1)
    fig.update_xaxes(showgrid=False, tickangle=-35)
    return _style(fig, height, legend=True)


def rubric_scorecard(metrics: pd.DataFrame, height: int = 300) -> go.Figure:
    fig = go.Figure([
        go.Bar(name="Qwen2.5-3B", x=metrics["metric"], y=metrics["Qwen2.5-3B"], marker=_bar_marker(VIOLET),
               text=[f"{v:.2f}" for v in metrics["Qwen2.5-3B"]], textposition="outside",
               textfont=dict(color=TEXT), cliponaxis=False,
               hovertemplate="%{x}<br>Qwen2.5-3B: %{y:.2f} / 5<extra></extra>"),
        go.Bar(name="Qwen3-4B", x=metrics["metric"], y=metrics["Qwen3-4B"], marker=_bar_marker(CYAN),
               text=[f"{v:.2f}" for v in metrics["Qwen3-4B"]], textposition="outside",
               textfont=dict(color=TEXT), cliponaxis=False,
               hovertemplate="%{x}<br>Qwen3-4B: %{y:.2f} / 5<extra></extra>"),
    ])
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Mean score (out of 5)", range=[0, 5.4], dtick=1)
    fig.update_xaxes(showgrid=False)
    return _style(fig, height, legend=True)


def qwen_delta_by_call(quality: pd.DataFrame, height: int = 340) -> go.Figure:
    df = quality.sort_values("improvement")
    colors = [CYAN if v > 0 else VIOLET if v < 0 else "#6b7488" for v in df["improvement"]]
    fig = go.Figure(go.Bar(
        y=df["call_id"], x=df["improvement"], orientation="h", marker=_bar_marker(colors),
        text=[f"{v:+.2f}" for v in df["improvement"]], textposition="outside",
        textfont=dict(color=TEXT), cliponaxis=False,
        hovertemplate="%{y}<br>Qwen3-4B − Qwen2.5-3B: %{x:+.2f} points<extra></extra>",
    ))
    span = max(df["improvement"].abs().max() if len(df) else 1, 0.5) * 1.35
    fig.update_xaxes(title="Score difference, Qwen3-4B minus Qwen2.5-3B (points on a 5-point scale)",
                     range=[-span, span], zeroline=True, zerolinewidth=1.5)
    fig.update_yaxes(showgrid=False)
    return _style(fig, height)
