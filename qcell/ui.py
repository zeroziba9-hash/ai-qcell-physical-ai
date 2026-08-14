"""Shared presentation primitives for the AI-QCell Streamlit application."""

from __future__ import annotations

from html import escape
from typing import Iterable, Mapping

import streamlit as st


GLOBAL_CSS = r"""
<style>
:root {
    --q-bg: #050913;
    --q-panel: rgba(12, 20, 36, 0.82);
    --q-panel-strong: #0c1424;
    --q-line: rgba(139, 160, 190, 0.18);
    --q-line-strong: rgba(86, 223, 255, 0.34);
    --q-text: #f6f9ff;
    --q-muted: #8fa2bb;
    --q-cyan: #55ddff;
    --q-blue: #5d7cff;
    --q-green: #55e5a3;
    --q-amber: #ffc866;
    --q-red: #ff6b7f;
    --q-radius: 18px;
    --q-shadow: 0 18px 60px rgba(0, 0, 0, 0.28);
}

html, body, [class*="css"] {
    font-family: Pretendard, SUIT, Inter, "Noto Sans KR", "Segoe UI", sans-serif;
}

[data-testid="stAppViewContainer"] {
    color: var(--q-text);
    background-color: var(--q-bg);
    background-image:
        radial-gradient(circle at 84% 2%, rgba(55, 102, 255, 0.17), transparent 26rem),
        radial-gradient(circle at 18% 28%, rgba(24, 197, 222, 0.08), transparent 24rem),
        linear-gradient(rgba(74, 110, 158, 0.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(74, 110, 158, 0.035) 1px, transparent 1px);
    background-size: auto, auto, 42px 42px, 42px 42px;
}

[data-testid="stHeader"] {
    height: 3.2rem;
    background: rgba(5, 9, 19, 0.72);
    border-bottom: 1px solid rgba(139, 160, 190, 0.08);
    backdrop-filter: blur(18px);
}

[data-testid="stMainBlockContainer"], .block-container {
    max-width: 1480px;
    padding-top: 1.45rem;
    padding-bottom: 4rem;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #07101e 0%, #070c16 100%);
    border-right: 1px solid var(--q-line);
}

section[data-testid="stSidebar"] > div:first-child::before {
    content: "AI · QCELL   /   OPS";
    display: block;
    margin: 0.9rem 1rem 0.55rem;
    padding: 0.78rem 0.9rem;
    color: var(--q-cyan);
    border: 1px solid rgba(85, 221, 255, 0.24);
    border-radius: 12px;
    background: linear-gradient(120deg, rgba(85, 221, 255, 0.09), rgba(93, 124, 255, 0.08));
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.14em;
}

[data-testid="stSidebarNav"] a {
    min-height: 2.6rem;
    margin: 0.12rem 0.52rem;
    padding: 0.48rem 0.72rem;
    color: #9fb0c6;
    border: 1px solid transparent;
    border-radius: 10px;
    transition: color 150ms ease, background 150ms ease, border-color 150ms ease;
}

[data-testid="stSidebarNav"] a:hover {
    color: var(--q-text);
    background: rgba(85, 221, 255, 0.06);
    border-color: rgba(85, 221, 255, 0.12);
}

[data-testid="stSidebarNav"] a[aria-current="page"] {
    color: white;
    background: linear-gradient(110deg, rgba(55, 181, 234, 0.18), rgba(93, 124, 255, 0.14));
    border-color: rgba(85, 221, 255, 0.26);
    box-shadow: inset 3px 0 0 var(--q-cyan);
}

h1, h2, h3, h4 {
    color: var(--q-text);
    letter-spacing: -0.025em;
}

h2 { margin-top: 0.35rem; }
h3 { font-size: 1.05rem; }
p, label, [data-testid="stCaptionContainer"] { color: var(--q-muted); }
hr { border-color: var(--q-line) !important; }

.qcell-hero {
    position: relative;
    overflow: hidden;
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    align-items: end;
    gap: 1.5rem;
    margin: 0 0 1.15rem;
    padding: clamp(1.3rem, 2.5vw, 2.1rem);
    border: 1px solid var(--q-line-strong);
    border-radius: 24px;
    background:
        linear-gradient(120deg, rgba(13, 28, 48, 0.96), rgba(9, 16, 31, 0.9)),
        var(--q-panel-strong);
    box-shadow: var(--q-shadow), inset 0 1px 0 rgba(255, 255, 255, 0.05);
}

.qcell-hero::before {
    content: "";
    position: absolute;
    inset: 0;
    pointer-events: none;
    background:
        linear-gradient(90deg, transparent 0 72%, rgba(85, 221, 255, 0.06) 72% 72.3%, transparent 72.3%),
        radial-gradient(circle at 89% 28%, rgba(85, 221, 255, 0.23), transparent 22%);
}

.qcell-hero-copy, .qcell-hero-meta { position: relative; z-index: 1; }
.qcell-kicker {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    margin-bottom: 0.62rem;
    color: var(--q-cyan);
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.16em;
    text-transform: uppercase;
}
.qcell-kicker::before {
    content: "";
    width: 1.9rem;
    height: 1px;
    background: var(--q-cyan);
    box-shadow: 0 0 12px rgba(85, 221, 255, 0.8);
}
.qcell-hero h1 {
    margin: 0;
    max-width: 900px;
    font-size: clamp(2rem, 4vw, 3.45rem);
    line-height: 1.02;
    font-weight: 850;
}
.qcell-hero p {
    max-width: 820px;
    margin: 0.82rem 0 0;
    color: #aebdd0;
    font-size: clamp(0.92rem, 1.3vw, 1.05rem);
    line-height: 1.7;
}
.qcell-hero-meta {
    min-width: 8.5rem;
    padding: 0.75rem 0.9rem;
    border: 1px solid rgba(85, 229, 163, 0.27);
    border-radius: 999px;
    color: #c8ffe5;
    background: rgba(26, 139, 91, 0.11);
    font-size: 0.7rem;
    font-weight: 800;
    letter-spacing: 0.11em;
    text-align: center;
    white-space: nowrap;
}
.qcell-hero-meta::before {
    content: "";
    display: inline-block;
    width: 0.46rem;
    height: 0.46rem;
    margin-right: 0.5rem;
    border-radius: 50%;
    background: var(--q-green);
    box-shadow: 0 0 0 4px rgba(85, 229, 163, 0.1), 0 0 13px rgba(85, 229, 163, 0.75);
}

.qcell-status-grid {
    display: grid;
    grid-template-columns: repeat(var(--q-count, 4), minmax(0, 1fr));
    gap: 0.68rem;
    margin: 0 0 1.15rem;
}
.qcell-status-item {
    min-width: 0;
    padding: 0.8rem 0.92rem;
    border: 1px solid var(--q-line);
    border-radius: 13px;
    background: rgba(9, 17, 31, 0.72);
}
.qcell-status-label {
    color: #7388a3;
    font-size: 0.66rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.qcell-status-value {
    overflow: hidden;
    margin-top: 0.28rem;
    color: #eaf3ff;
    font-size: 0.88rem;
    font-weight: 700;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.qcell-status-value[data-tone="good"] { color: var(--q-green); }
.qcell-status-value[data-tone="warn"] { color: var(--q-amber); }
.qcell-status-value[data-tone="bad"] { color: var(--q-red); }

.qcell-flow {
    display: grid;
    grid-template-columns: repeat(var(--q-count, 4), minmax(0, 1fr));
    gap: 0.45rem;
    margin: 0 0 1.45rem;
}
.qcell-flow-step {
    position: relative;
    display: flex;
    align-items: center;
    gap: 0.68rem;
    min-height: 3.15rem;
    padding: 0.68rem 0.82rem;
    border: 1px solid var(--q-line);
    border-radius: 12px;
    background: rgba(8, 16, 29, 0.7);
}
.qcell-flow-step:not(:last-child)::after {
    content: "";
    position: absolute;
    top: 50%;
    right: -0.52rem;
    z-index: 2;
    width: 0.56rem;
    height: 1px;
    background: var(--q-cyan);
}
.qcell-flow-index {
    display: grid;
    flex: 0 0 auto;
    width: 1.65rem;
    height: 1.65rem;
    place-items: center;
    border-radius: 8px;
    color: var(--q-cyan);
    background: rgba(85, 221, 255, 0.1);
    font-size: 0.67rem;
    font-weight: 800;
}
.qcell-flow-label { color: #dce9f8; font-size: 0.78rem; font-weight: 700; }

.qcell-section {
    display: flex;
    align-items: end;
    justify-content: space-between;
    gap: 1rem;
    margin: 1.75rem 0 0.8rem;
}
.qcell-section h2 { margin: 0; font-size: clamp(1.22rem, 2vw, 1.62rem); }
.qcell-section p { max-width: 720px; margin: 0.28rem 0 0; font-size: 0.83rem; line-height: 1.55; }
.qcell-section-code { color: #657d9b; font: 700 0.68rem/1.2 Consolas, monospace; letter-spacing: 0.08em; }

[data-testid="stMetric"] {
    min-height: 7.2rem;
    padding: 1rem 1.05rem;
    border: 1px solid var(--q-line) !important;
    border-radius: var(--q-radius) !important;
    background: linear-gradient(145deg, rgba(15, 27, 47, 0.9), rgba(8, 15, 28, 0.9)) !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.035), 0 10px 30px rgba(0,0,0,0.14);
}
[data-testid="stMetric"] label { color: #8094ad !important; font-size: 0.75rem !important; }
[data-testid="stMetricValue"] { color: #f5f9ff; font-weight: 800; letter-spacing: -0.035em; }
[data-testid="stMetricDelta"] { font-weight: 700; }

[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: var(--q-line) !important;
    border-radius: var(--q-radius) !important;
    background: rgba(8, 16, 29, 0.55);
    box-shadow: 0 12px 32px rgba(0,0,0,0.12);
}

.stButton > button, .stDownloadButton > button, [data-testid="stPageLink"] a {
    min-height: 2.75rem;
    border: 1px solid rgba(129, 154, 188, 0.26) !important;
    border-radius: 11px !important;
    color: #e8f2ff !important;
    background: linear-gradient(145deg, rgba(22, 35, 56, 0.94), rgba(12, 21, 38, 0.94)) !important;
    font-weight: 750 !important;
    transition: transform 150ms ease, border-color 150ms ease, box-shadow 150ms ease !important;
}
.stButton > button:hover, .stDownloadButton > button:hover, [data-testid="stPageLink"] a:hover {
    transform: translateY(-1px);
    border-color: rgba(85, 221, 255, 0.55) !important;
    box-shadow: 0 8px 24px rgba(45, 190, 231, 0.11);
}
.stButton > button[kind="primary"], button[data-testid="stBaseButton-primary"] {
    border-color: rgba(85, 221, 255, 0.65) !important;
    color: #03111c !important;
    background: linear-gradient(110deg, #55ddff, #6e8cff) !important;
    box-shadow: 0 8px 26px rgba(75, 169, 255, 0.24);
}

[data-baseweb="select"] > div,
[data-baseweb="input"] > div,
[data-testid="stFileUploaderDropzone"],
[data-testid="stTextInputRootElement"] {
    border-color: rgba(129, 154, 188, 0.24) !important;
    border-radius: 11px !important;
    background: rgba(7, 14, 26, 0.82) !important;
}
[data-testid="stFileUploaderDropzone"] { padding: 1rem; }

[data-testid="stTabs"] [role="tablist"] {
    gap: 0.35rem;
    padding: 0.3rem;
    border: 1px solid var(--q-line);
    border-radius: 12px;
    background: rgba(8, 16, 29, 0.72);
}
[data-testid="stTabs"] button[role="tab"] { border-radius: 9px; }
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: white;
    background: rgba(85, 221, 255, 0.09);
}

[data-testid="stDataFrame"], [data-testid="stTable"] {
    overflow: hidden;
    border: 1px solid var(--q-line);
    border-radius: 14px;
    background: rgba(7, 14, 26, 0.78);
}
[data-testid="stImage"] img { border-radius: 14px; border: 1px solid var(--q-line); }
[data-testid="stAlert"] { border-radius: 13px; border-color: var(--q-line); }
[data-testid="stExpander"] { border-color: var(--q-line); border-radius: 13px; background: rgba(8, 16, 29, 0.65); }
[data-testid="stProgress"] > div > div { background: linear-gradient(90deg, var(--q-cyan), var(--q-blue)); }

.qcell-module-card {
    min-height: 9.2rem;
    margin-bottom: 0.6rem;
    padding: 1.05rem;
    border: 1px solid var(--q-line);
    border-radius: var(--q-radius);
    background: linear-gradient(145deg, rgba(15, 27, 47, 0.84), rgba(7, 14, 26, 0.82));
}
.qcell-module-card span { color: var(--q-cyan); font: 800 0.65rem/1 Consolas, monospace; letter-spacing: 0.1em; }
.qcell-module-card h3 { margin: 0.72rem 0 0.42rem; font-size: 1.03rem; }
.qcell-module-card p { margin: 0; font-size: 0.79rem; line-height: 1.55; }

*:focus-visible {
    outline: 2px solid var(--q-cyan) !important;
    outline-offset: 2px !important;
}

@media (max-width: 900px) {
    [data-testid="stMainBlockContainer"], .block-container { padding-left: 1rem; padding-right: 1rem; }
    .qcell-hero { grid-template-columns: 1fr; align-items: start; }
    .qcell-hero-meta { justify-self: start; }
    .qcell-status-grid, .qcell-flow { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .qcell-flow-step:nth-child(2)::after { display: none; }
}

@media (max-width: 560px) {
    .qcell-hero { padding: 1.15rem; border-radius: 18px; }
    .qcell-hero h1 { font-size: 1.88rem; }
    .qcell-status-grid, .qcell-flow { grid-template-columns: 1fr; }
    .qcell-flow-step::after { display: none; }
    .qcell-section { align-items: start; flex-direction: column; }
}

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { scroll-behavior: auto !important; transition: none !important; animation: none !important; }
}
</style>
"""


def inject_global_css() -> None:
    """Apply the app-wide AI-QCell visual language after any page-local CSS."""

    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def page_header(
    eyebrow: str,
    title: str,
    description: str,
    *,
    status: str = "SYSTEM ONLINE",
) -> None:
    """Render a consistent page hero with safely escaped copy."""

    st.markdown(
        page_header_html(eyebrow, title, description, status=status),
        unsafe_allow_html=True,
    )


def page_header_html(
    eyebrow: str,
    title: str,
    description: str,
    *,
    status: str = "SYSTEM ONLINE",
) -> str:
    return (
        '<div class="qcell-hero">'
        '<div class="qcell-hero-copy">'
        f'<div class="qcell-kicker">{escape(eyebrow)}</div>'
        f'<h1>{escape(title)}</h1>'
        f'<p>{escape(description)}</p>'
        '</div>'
        f'<div class="qcell-hero-meta">{escape(status)}</div>'
        '</div>'
    )


def status_strip(items: Iterable[Mapping[str, str]]) -> None:
    """Render compact operational status values above dense content."""

    materialized = list(items)
    cards = "".join(
        '<div class="qcell-status-item">'
        f'<div class="qcell-status-label">{escape(str(item["label"]))}</div>'
        f'<div class="qcell-status-value" data-tone="{escape(str(item.get("tone", "neutral")))}">'
        f'{escape(str(item["value"]))}</div>'
        '</div>'
        for item in materialized
    )
    st.markdown(
        f'<div class="qcell-status-grid" style="--q-count:{max(1, len(materialized))}">{cards}</div>',
        unsafe_allow_html=True,
    )


def workflow_strip(steps: Iterable[str]) -> None:
    """Show the physical-AI loop as a lightweight numbered process strip."""

    materialized = list(steps)
    cards = "".join(
        '<div class="qcell-flow-step">'
        f'<div class="qcell-flow-index">{index:02d}</div>'
        f'<div class="qcell-flow-label">{escape(step)}</div>'
        '</div>'
        for index, step in enumerate(materialized, start=1)
    )
    st.markdown(
        f'<div class="qcell-flow" style="--q-count:{max(1, len(materialized))}">{cards}</div>',
        unsafe_allow_html=True,
    )


def section_header(title: str, description: str = "", *, code: str = "") -> None:
    """Separate dashboard sections without using oversized Streamlit headings."""

    description_html = f'<p>{escape(description)}</p>' if description else ""
    code_html = f'<div class="qcell-section-code">{escape(code)}</div>' if code else ""
    st.markdown(
        '<div class="qcell-section">'
        f'<div><h2>{escape(title)}</h2>{description_html}</div>'
        f'{code_html}'
        '</div>',
        unsafe_allow_html=True,
    )


def module_card(code: str, title: str, description: str) -> None:
    """Introduce a linked application module in the home launchpad."""

    st.markdown(
        '<div class="qcell-module-card">'
        f'<span>{escape(code)}</span>'
        f'<h3>{escape(title)}</h3>'
        f'<p>{escape(description)}</p>'
        '</div>',
        unsafe_allow_html=True,
    )
