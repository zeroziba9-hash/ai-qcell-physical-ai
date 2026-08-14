"""Product-grade presentation primitives for the AI-QCell Streamlit console."""

from __future__ import annotations

from html import escape
import inspect
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import streamlit as st


GLOBAL_CSS = r"""
<style>
:root {
    --q-bg: #080b10;
    --q-sidebar: #090d13;
    --q-panel: #0f141c;
    --q-panel-raised: #121923;
    --q-panel-soft: #0c1118;
    --q-line: #222c38;
    --q-line-soft: rgba(125, 145, 168, 0.13);
    --q-text: #eaf0f5;
    --q-muted: #8796a8;
    --q-dim: #566477;
    --q-accent: #5be0b8;
    --q-cyan: #5be0b8;
    --q-blue: #6aa7ff;
    --q-amber: #f5bd62;
    --q-red: #ff7687;
    --q-radius: 12px;
    --q-shadow: 0 20px 55px rgba(0, 0, 0, 0.22);
}

html, body, [class*="css"] {
    font-family: Pretendard, SUIT, Inter, "Noto Sans KR", "Segoe UI", sans-serif;
    font-variant-numeric: tabular-nums;
}

[data-testid="stAppViewContainer"] {
    color: var(--q-text);
    background-color: var(--q-bg);
    background-image:
        linear-gradient(rgba(111, 132, 157, 0.022) 1px, transparent 1px),
        linear-gradient(90deg, rgba(111, 132, 157, 0.022) 1px, transparent 1px),
        radial-gradient(circle at 77% -8%, rgba(91, 224, 184, 0.065), transparent 30rem);
    background-size: 64px 64px, 64px 64px, auto;
}

[data-testid="stHeader"] {
    height: 0;
    min-height: 0;
    background: transparent;
    border: 0;
}
[data-testid="stToolbar"], [data-testid="stAppDeployButton"] { display: none !important; }
[data-testid="stDecoration"] { display: none; }

[data-testid="stMainBlockContainer"], .block-container {
    width: 100%;
    max-width: 1540px;
    padding-top: clamp(0.8rem, 1.4vw, 1.25rem) !important;
    padding-right: clamp(1rem, 3vw, 2.5rem) !important;
    padding-bottom: 4rem !important;
    padding-left: clamp(1rem, 3vw, 2.5rem) !important;
}

section[data-testid="stSidebar"] {
    width: 17.25rem !important;
    background: var(--q-sidebar);
    border-right: 1px solid var(--q-line);
}
section[data-testid="stSidebar"] > div:first-child { padding-top: 0.55rem; }

.qcell-side-brand {
    display: grid;
    grid-template-columns: 2.45rem minmax(0, 1fr) auto;
    align-items: center;
    gap: 0.72rem;
    margin: 0.25rem 0.25rem 1rem;
    padding: 0.72rem;
    border: 1px solid var(--q-line);
    border-radius: 10px;
    background: #0d131b;
}
.qcell-side-mark {
    display: grid;
    width: 2.45rem;
    height: 2.45rem;
    place-items: center;
    color: #07100d;
    border-radius: 8px;
    background: var(--q-accent);
    font: 900 0.88rem/1 Inter, sans-serif;
    letter-spacing: -0.05em;
}
.qcell-side-name { color: #f3f7fa; font-size: 0.84rem; font-weight: 820; letter-spacing: 0.02em; }
.qcell-side-sub { margin-top: 0.12rem; color: var(--q-dim); font: 600 0.59rem/1.2 Consolas, monospace; letter-spacing: 0.1em; }
.qcell-side-live { color: var(--q-accent); font: 800 0.58rem/1 Consolas, monospace; letter-spacing: 0.09em; }
.qcell-side-live::before {
    content: "";
    display: inline-block;
    width: 0.36rem;
    height: 0.36rem;
    margin-right: 0.35rem;
    border-radius: 50%;
    background: var(--q-accent);
    box-shadow: 0 0 9px rgba(91, 224, 184, 0.55);
}
.qcell-nav-group {
    margin: 1rem 0.68rem 0.35rem;
    color: #4f5d6e;
    font: 750 0.58rem/1.2 Consolas, monospace;
    letter-spacing: 0.15em;
}
section[data-testid="stSidebar"] [data-testid="stPageLink"] { margin: 0.08rem 0.25rem; }
section[data-testid="stSidebar"] [data-testid="stPageLink"] a {
    min-height: 2.25rem;
    padding: 0.42rem 0.7rem !important;
    color: #8e9caf !important;
    border: 1px solid transparent !important;
    border-radius: 8px !important;
    background: transparent !important;
    box-shadow: none !important;
    font-size: 0.76rem !important;
    font-weight: 650 !important;
    transition: color 130ms ease, background 130ms ease, border-color 130ms ease !important;
}
section[data-testid="stSidebar"] [data-testid="stPageLink"] a:hover {
    transform: none !important;
    color: #dce5ed !important;
    border-color: rgba(125, 145, 168, 0.14) !important;
    background: rgba(125, 145, 168, 0.055) !important;
}
section[data-testid="stSidebar"] [data-testid="stPageLink"] a[aria-current="page"] {
    color: #f2f7f5 !important;
    border-color: rgba(91, 224, 184, 0.18) !important;
    background: rgba(91, 224, 184, 0.075) !important;
    box-shadow: inset 2px 0 0 var(--q-accent) !important;
}
.qcell-nav-link {
    display: grid;
    grid-template-columns: 1.35rem minmax(0, 1fr);
    align-items: center;
    gap: 0.48rem;
    min-height: 2.25rem;
    margin: 0.08rem 0.25rem;
    padding: 0.42rem 0.7rem;
    color: #8e9caf !important;
    border: 1px solid transparent;
    border-radius: 8px;
    background: transparent;
    text-decoration: none !important;
    transition: color 130ms ease, background 130ms ease, border-color 130ms ease;
}
.qcell-nav-link:hover {
    color: #dce5ed !important;
    border-color: rgba(125, 145, 168, 0.14);
    background: rgba(125, 145, 168, 0.055);
}
.qcell-nav-link.is-active {
    color: #f2f7f5 !important;
    border-color: rgba(91, 224, 184, 0.18);
    background: rgba(91, 224, 184, 0.075);
    box-shadow: inset 2px 0 0 var(--q-accent);
}
.qcell-nav-index {
    color: #475568;
    font: 700 0.56rem/1 Consolas, monospace;
}
.qcell-nav-link.is-active .qcell-nav-index { color: var(--q-accent); }
.qcell-nav-title { overflow: hidden; font-size: 0.74rem; font-weight: 650; text-overflow: ellipsis; white-space: nowrap; }
.qcell-side-foot {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: 1.1rem 0.5rem 0.7rem;
    padding-top: 0.75rem;
    color: #526071;
    border-top: 1px solid var(--q-line-soft);
    font: 600 0.58rem/1.4 Consolas, monospace;
}
.qcell-side-foot b { color: #8fa095; font-weight: 700; }

h1, h2, h3, h4 { color: var(--q-text); letter-spacing: -0.025em; }
h3 { font-size: 1rem; }
p, label, [data-testid="stCaptionContainer"] { color: var(--q-muted); }
hr { border-color: var(--q-line) !important; }

.qcell-hero {
    position: relative;
    overflow: hidden;
    display: grid;
    grid-template-columns: minmax(0, 1fr) 12.5rem;
    align-items: center;
    gap: 2rem;
    margin: 0 0 0.85rem;
    padding: clamp(1.2rem, 2.2vw, 1.75rem);
    border: 1px solid var(--q-line);
    border-radius: 14px;
    background: linear-gradient(115deg, #111821, #0d131b 76%);
    box-shadow: var(--q-shadow), inset 0 1px 0 rgba(255, 255, 255, 0.025);
}
.qcell-hero::before {
    content: "";
    position: absolute;
    inset: 0 auto 0 0;
    width: 3px;
    background: linear-gradient(180deg, var(--q-accent), rgba(91, 224, 184, 0.08));
}
.qcell-hero::after {
    content: "";
    position: absolute;
    inset: 0 0 0 auto;
    width: 38%;
    height: 100%;
    border: 0;
    border-radius: 0;
    background:
        linear-gradient(90deg, transparent, rgba(91, 224, 184, 0.028)),
        repeating-linear-gradient(135deg, transparent 0 24px, rgba(91, 224, 184, 0.035) 24px 25px);
    box-shadow: none;
    opacity: 0.62;
    mask-image: linear-gradient(90deg, transparent, black 52%);
}
.qcell-hero-copy, .qcell-hero-meta { position: relative; z-index: 1; }
.qcell-kicker {
    display: flex;
    align-items: center;
    gap: 0.58rem;
    margin-bottom: 0.58rem;
    color: var(--q-accent);
    font: 750 0.63rem/1.2 Consolas, monospace;
    letter-spacing: 0.14em;
    text-transform: uppercase;
}
.qcell-kicker::before { content: "//"; color: #3b755f; }
.qcell-hero h1 {
    margin: 0;
    max-width: 900px;
    font-size: clamp(1.85rem, 3.4vw, 2.85rem);
    line-height: 1.08;
    font-weight: 830;
}
.qcell-hero p {
    max-width: 800px;
    margin: 0.68rem 0 0;
    color: #98a7b7;
    font-size: clamp(0.83rem, 1.15vw, 0.94rem);
    line-height: 1.65;
    word-break: keep-all;
    overflow-wrap: break-word;
}
.qcell-hero-meta {
    padding: 0.82rem 0.9rem;
    color: #aab8c5;
    border-left: 1px solid var(--q-line);
    font: 750 0.62rem/1.45 Consolas, monospace;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.qcell-hero-meta::before {
    content: "STATUS";
    display: block;
    margin-bottom: 0.28rem;
    color: #4e5c6e;
    font-size: 0.52rem;
    letter-spacing: 0.15em;
}
.qcell-hero-meta::after {
    content: "";
    display: inline-block;
    width: 0.4rem;
    height: 0.4rem;
    margin-left: 0.52rem;
    border-radius: 50%;
    background: var(--q-accent);
    box-shadow: 0 0 9px rgba(91, 224, 184, 0.6);
}

.qcell-status-grid {
    display: grid;
    grid-template-columns: repeat(var(--q-count, 4), minmax(0, 1fr));
    margin: 0 0 0.75rem;
    border: 1px solid var(--q-line);
    border-radius: 10px;
    background: var(--q-panel-soft);
}
.qcell-status-item { min-width: 0; padding: 0.72rem 0.9rem; }
.qcell-status-item + .qcell-status-item { border-left: 1px solid var(--q-line); }
.qcell-status-label {
    color: #536174;
    font: 700 0.55rem/1.2 Consolas, monospace;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}
.qcell-status-value {
    overflow: hidden;
    margin-top: 0.3rem;
    color: #d7e0e8;
    font-size: 0.79rem;
    font-weight: 720;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.qcell-status-value[data-tone="good"] { color: var(--q-accent); }
.qcell-status-value[data-tone="warn"] { color: var(--q-amber); }
.qcell-status-value[data-tone="bad"] { color: var(--q-red); }

.qcell-flow {
    display: grid;
    grid-template-columns: repeat(var(--q-count, 4), minmax(0, 1fr));
    margin: 0 0 1.25rem;
    padding: 0.58rem 0;
    border-top: 1px solid var(--q-line-soft);
    border-bottom: 1px solid var(--q-line-soft);
}
.qcell-flow-step { position: relative; display: flex; align-items: center; gap: 0.6rem; padding: 0.18rem 0.8rem; }
.qcell-flow-step + .qcell-flow-step { border-left: 1px solid var(--q-line-soft); }
.qcell-flow-index { color: var(--q-accent); font: 750 0.59rem/1 Consolas, monospace; }
.qcell-flow-label { color: #7f8fa1; font-size: 0.72rem; font-weight: 660; }

.qcell-section {
    display: flex;
    align-items: end;
    justify-content: space-between;
    gap: 1rem;
    margin: 1.55rem 0 0.7rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid var(--q-line-soft);
}
.qcell-section h2 { margin: 0; font-size: clamp(1.1rem, 1.8vw, 1.4rem); font-weight: 780; }
.qcell-section p { max-width: 720px; margin: 0.25rem 0 0; color: #708095; font-size: 0.77rem; line-height: 1.5; }
.qcell-section-code { color: #465466; font: 700 0.58rem/1.2 Consolas, monospace; letter-spacing: 0.1em; }

[data-testid="stMetric"] {
    position: relative;
    min-height: 6.9rem;
    padding: 0.92rem 1rem;
    border: 1px solid var(--q-line) !important;
    border-radius: var(--q-radius) !important;
    background: var(--q-panel) !important;
    box-shadow: none !important;
}
[data-testid="stMetric"]::before {
    content: "";
    position: absolute;
    top: -1px;
    left: 1rem;
    width: 2.2rem;
    height: 2px;
    background: var(--q-accent);
}
[data-testid="stMetric"] label { color: #66768a !important; font-size: 0.69rem !important; }
[data-testid="stMetricValue"] {
    overflow: visible !important;
    color: #edf3f6;
    font-size: clamp(1.18rem, 2.15vw, 1.9rem) !important;
    font-weight: 790;
    line-height: 1.08 !important;
    letter-spacing: -0.035em;
    text-overflow: clip !important;
    white-space: normal !important;
    word-break: keep-all;
    overflow-wrap: anywhere;
}
[data-testid="stMetricValue"] > div {
    overflow: visible !important;
    text-overflow: clip !important;
    white-space: normal !important;
}
[data-testid="stMetricValue"] p {
    margin: 0 !important;
    overflow: visible !important;
    text-overflow: clip !important;
    white-space: normal !important;
    word-break: keep-all;
    overflow-wrap: anywhere;
}
[data-testid="stMetricDelta"] { font-size: 0.65rem; font-weight: 700; }
[data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"]:nth-child(5):last-child [data-testid="stMetric"]) {
    display: grid !important;
    grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)) !important;
    gap: 0.65rem !important;
}
[data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"]:nth-child(5):last-child [data-testid="stMetric"]) > [data-testid="stColumn"] {
    width: auto !important;
    min-width: 0 !important;
    flex: none !important;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: var(--q-line) !important;
    border-radius: var(--q-radius) !important;
    background: rgba(15, 20, 28, 0.7);
    box-shadow: none;
}

.stButton > button, .stDownloadButton > button {
    min-height: 2.55rem;
    border: 1px solid #2a3542 !important;
    border-radius: 8px !important;
    color: #d9e2e9 !important;
    background: #121923 !important;
    box-shadow: none !important;
    font-weight: 720 !important;
    transition: border-color 130ms ease, background 130ms ease !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    transform: none !important;
    border-color: #526476 !important;
    background: #161e29 !important;
}
.stButton > button[kind="primary"], button[data-testid="stBaseButton-primary"] {
    color: #07120e !important;
    border-color: var(--q-accent) !important;
    background: var(--q-accent) !important;
}

[data-baseweb="select"] > div, [data-baseweb="input"] > div,
[data-testid="stFileUploaderDropzone"], [data-testid="stTextInputRootElement"] {
    border-color: #283341 !important;
    border-radius: 8px !important;
    background: #0c1118 !important;
}
[data-testid="stFileUploaderDropzone"] { padding: 0.9rem; }
[data-testid="stTabs"] [role="tablist"] { gap: 0.2rem; border-bottom: 1px solid var(--q-line); }
[data-testid="stTabs"] button[role="tab"] { border-radius: 7px 7px 0 0; }
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] { color: var(--q-accent); background: rgba(91, 224, 184, 0.045); }
[data-testid="stDataFrame"], [data-testid="stTable"] { overflow: hidden; border: 1px solid var(--q-line); border-radius: 9px; }
[data-testid="stImage"] img { border: 1px solid var(--q-line); border-radius: 9px; }
[data-testid="stAlert"] { border-radius: 8px; border-color: var(--q-line); }
[data-testid="stExpander"] { border-color: var(--q-line); border-radius: 8px; background: #0d1219; }
[data-testid="stProgress"] > div > div { background: var(--q-accent); }

.qcell-launch-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.7rem;
}
.qcell-launch-card {
    position: relative;
    overflow: hidden;
    min-height: 10.3rem;
    padding: 1rem;
    color: inherit !important;
    border: 1px solid var(--q-line);
    border-radius: 11px;
    background: var(--q-panel);
    text-decoration: none !important;
    transition: border-color 140ms ease, background 140ms ease, transform 140ms ease;
}
.qcell-launch-card:hover {
    transform: translateY(-2px);
    border-color: #3b4a59;
    background: var(--q-panel-raised);
}
.qcell-launch-card[data-featured="true"] {
    grid-column: span 2;
    background: linear-gradient(120deg, #101a20, #10161f);
}
.qcell-launch-card[data-featured="true"]::after {
    content: "";
    position: absolute;
    right: -3rem;
    bottom: -4rem;
    width: 11rem;
    height: 11rem;
    border: 1px solid rgba(91, 224, 184, 0.09);
    border-radius: 50%;
    box-shadow: 0 0 0 2rem rgba(91, 224, 184, 0.02), 0 0 0 4rem rgba(91, 224, 184, 0.012);
}
.qcell-launch-top { display: flex; align-items: center; justify-content: space-between; gap: 0.5rem; }
.qcell-launch-code { color: #526173; font: 700 0.55rem/1.2 Consolas, monospace; letter-spacing: 0.1em; }
.qcell-launch-arrow { color: #536274; font-size: 1rem; transition: color 140ms ease, transform 140ms ease; }
.qcell-launch-card:hover .qcell-launch-arrow { color: var(--q-accent); transform: translateX(2px); }
.qcell-launch-card h3 { margin: 1.15rem 0 0.42rem; color: #e5ecef; font-size: 0.95rem; font-weight: 750; }
.qcell-launch-card p {
    max-width: 30rem; margin: 0; color: #718095; font-size: 0.72rem; line-height: 1.55; word-break: keep-all; overflow-wrap: break-word;
}
.qcell-launch-tag {
    display: inline-block;
    margin-top: 0.9rem;
    color: #6d817a;
    font: 650 0.54rem/1.2 Consolas, monospace;
    letter-spacing: 0.09em;
}

*:focus-visible { outline: 2px solid var(--q-accent) !important; outline-offset: 2px !important; }

@media (max-width: 1080px) {
    .qcell-launch-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 900px) {
    [data-testid="stHeader"] {
        height: 2.5rem;
        min-height: 2.5rem;
        background: rgba(8, 11, 16, 0.94);
        border-bottom: 1px solid rgba(125, 145, 168, 0.07);
        backdrop-filter: blur(16px);
    }
    [data-testid="stMainBlockContainer"], .block-container { padding-top: 0.8rem !important; padding-left: 1rem !important; padding-right: 1rem !important; }
    .qcell-hero { grid-template-columns: 1fr; gap: 1rem; }
    .qcell-hero-meta { border-top: 1px solid var(--q-line); border-left: 0; }
    .qcell-status-grid, .qcell-flow { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .qcell-status-item:nth-child(3) { border-left: 0; border-top: 1px solid var(--q-line); }
    .qcell-status-item:nth-child(4) { border-top: 1px solid var(--q-line); }
}
@media (max-width: 560px) {
    .qcell-hero { padding: 1.05rem; border-radius: 10px; }
    .qcell-hero h1 { font-size: 1.72rem; }
    .qcell-status-grid, .qcell-flow, .qcell-launch-grid { grid-template-columns: 1fr; }
    .qcell-launch-card[data-featured="true"] { grid-column: span 1; }
    .qcell-status-item + .qcell-status-item { border-top: 1px solid var(--q-line); border-left: 0; }
    .qcell-flow-step + .qcell-flow-step { border-top: 1px solid var(--q-line-soft); border-left: 0; }
    .qcell-section { align-items: start; flex-direction: column; }
}
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { scroll-behavior: auto !important; transition: none !important; animation: none !important; }
}
</style>
"""


NAV_GROUPS: Sequence[tuple[str, Sequence[tuple[str, str]]]] = (
    ("OVERVIEW", (("app.py", "00  운영 대시보드"),)),
    (
        "INSPECTION",
        (
            ("pages/1_vision_inspection.py", "01  기준 영상 검사"),
            ("pages/2_trained_patch_model.py", "02  Patch Memory"),
            ("pages/3_deep_patchcore_mvtec.py", "03  Deep PatchCore"),
            ("pages/5_realtime_inspection.py", "04  실시간 검사"),
        ),
    ),
    (
        "AUTOMATION",
        (
            ("pages/4_ros2_sorting_pipeline.py", "05  ROS2 자동 선별"),
            ("pages/6_actuator_digital_twin.py", "06  액추에이터 트윈"),
            ("pages/11_edge_runtime_benchmark.py", "07  Edge Runtime"),
        ),
    ),
    (
        "MODEL LIFECYCLE",
        (
            ("pages/7_dataset_studio.py", "08  Dataset Studio"),
            ("pages/8_training_lab.py", "09  Training Lab"),
            ("pages/9_model_registry.py", "10  Model Registry"),
            ("pages/10_review_queue.py", "11  Review Queue"),
        ),
    ),
)


def _render_sidebar_navigation(active_script: str) -> None:
    st.sidebar.markdown(
        """
        <div class="qcell-side-brand">
          <div class="qcell-side-mark">QC</div>
          <div><div class="qcell-side-name">AI-QCell</div><div class="qcell-side-sub">QUALITY OS / R2</div></div>
          <div class="qcell-side-live">LIVE</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    for group, links in NAV_GROUPS:
        rendered_links = []
        for page, label in links:
            page_name = page.rsplit("/", 1)[-1]
            stem = page_name.removesuffix(".py")
            href = "/" if stem == "app" else f'/{stem.split("_", 1)[-1]}'
            index, title = label.split("  ", 1)
            active_class = " is-active" if page_name == active_script else ""
            aria_current = ' aria-current="page"' if active_class else ""
            rendered_links.append(
                f'<a class="qcell-nav-link{active_class}" href="{escape(href, quote=True)}" '
                f'target="_self"{aria_current}><span class="qcell-nav-index">{escape(index)}</span>'
                f'<span class="qcell-nav-title">{escape(title)}</span></a>'
            )
        st.sidebar.markdown(
            f'<div class="qcell-nav-group">{escape(group)}</div>{"".join(rendered_links)}',
            unsafe_allow_html=True,
        )
    st.sidebar.markdown(
        '<div class="qcell-side-foot"><span>EDGE NODE / 01</span><b>CONNECTED</b></div>',
        unsafe_allow_html=True,
    )


def inject_global_css() -> None:
    """Apply the visual system early and render the grouped product navigation."""

    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    caller = Path(inspect.stack()[1].filename).name
    _render_sidebar_navigation(caller)


def page_header(eyebrow: str, title: str, description: str, *, status: str = "SYSTEM ONLINE") -> None:
    st.markdown(page_header_html(eyebrow, title, description, status=status), unsafe_allow_html=True)


def page_header_html(eyebrow: str, title: str, description: str, *, status: str = "SYSTEM ONLINE") -> str:
    return (
        '<div class="qcell-hero"><div class="qcell-hero-copy">'
        f'<div class="qcell-kicker">{escape(eyebrow)}</div><h1>{escape(title)}</h1>'
        f'<p>{escape(description)}</p></div><div class="qcell-hero-meta">{escape(status)}</div></div>'
    )


def status_strip(items: Iterable[Mapping[str, str]]) -> None:
    materialized = list(items)
    cards = "".join(
        '<div class="qcell-status-item">'
        f'<div class="qcell-status-label">{escape(str(item["label"]))}</div>'
        f'<div class="qcell-status-value" data-tone="{escape(str(item.get("tone", "neutral")))}">'
        f'{escape(str(item["value"]))}</div></div>'
        for item in materialized
    )
    st.markdown(
        f'<div class="qcell-status-grid" style="--q-count:{max(1, len(materialized))}">{cards}</div>',
        unsafe_allow_html=True,
    )


def workflow_strip(steps: Iterable[str]) -> None:
    materialized = list(steps)
    cards = "".join(
        f'<div class="qcell-flow-step"><div class="qcell-flow-index">{index:02d}</div>'
        f'<div class="qcell-flow-label">{escape(step)}</div></div>'
        for index, step in enumerate(materialized, start=1)
    )
    st.markdown(
        f'<div class="qcell-flow" style="--q-count:{max(1, len(materialized))}">{cards}</div>',
        unsafe_allow_html=True,
    )


def section_header(title: str, description: str = "", *, code: str = "") -> None:
    description_html = f'<p>{escape(description)}</p>' if description else ""
    code_html = f'<div class="qcell-section-code">{escape(code)}</div>' if code else ""
    st.markdown(
        f'<div class="qcell-section"><div><h2>{escape(title)}</h2>{description_html}</div>{code_html}</div>',
        unsafe_allow_html=True,
    )


def module_grid(items: Iterable[Mapping[str, str | bool]]) -> None:
    cards = []
    for item in items:
        featured = "true" if bool(item.get("featured", False)) else "false"
        cards.append(
            f'<a class="qcell-launch-card" data-featured="{featured}" '
            f'href="{escape(str(item["href"]), quote=True)}" target="_self">'
            '<div class="qcell-launch-top">'
            f'<span class="qcell-launch-code">{escape(str(item["code"]))}</span>'
            '<span class="qcell-launch-arrow">→</span></div>'
            f'<h3>{escape(str(item["title"]))}</h3>'
            f'<p>{escape(str(item["description"]))}</p>'
            f'<span class="qcell-launch-tag">{escape(str(item.get("tag", "OPEN MODULE")))}</span>'
            '</a>'
        )
    st.markdown(f'<div class="qcell-launch-grid">{"".join(cards)}</div>', unsafe_allow_html=True)
