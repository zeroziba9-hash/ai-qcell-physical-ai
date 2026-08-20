from pathlib import Path

from qcell.ui import GLOBAL_CSS, NAV_GROUPS, page_header_html


def test_page_header_escapes_untrusted_copy() -> None:
    rendered = page_header_html(
        "<script>alert(1)</script>",
        "Inspection & Control",
        'A < B and "quoted"',
        status="READY > IDLE",
    )

    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
    assert "Inspection &amp; Control" in rendered
    assert "A &lt; B" in rendered
    assert "READY &gt; IDLE" in rendered


def test_design_system_includes_responsive_and_accessible_states() -> None:
    assert "@media (max-width: 560px)" in GLOBAL_CSS
    assert "@media (prefers-reduced-motion: reduce)" in GLOBAL_CSS
    assert ":focus-visible" in GLOBAL_CSS
    assert "--q-cyan" in GLOBAL_CSS
    assert '[data-testid="stPageLink"] a[aria-current="page"]' in GLOBAL_CSS
    assert '[data-testid="stHeader"]' in GLOBAL_CSS
    assert "text-overflow: clip !important" in GLOBAL_CSS
    assert ':has(> [data-testid="stColumn"]:nth-child(5):last-child' in GLOBAL_CSS
    assert '[data-testid="stMetricValue"] p' in GLOBAL_CSS
    assert "repeat(auto-fit, minmax(190px, 1fr))" in GLOBAL_CSS


def test_navigation_covers_every_operational_page_once() -> None:
    links = [link for _, group_links in NAV_GROUPS for link in group_links]
    pages = [page for page, _ in links]

    assert len(NAV_GROUPS) == 5
    assert len(pages) == 15
    assert len(set(pages)) == len(pages)
    assert pages[0] == "app.py"
    assert "pages/12_quality_analytics.py" in pages
    assert "pages/13_access_control.py" in pages
    assert "pages/14_traceability.py" in pages
    assert "pages/3_deep_patchcore_mvtec.py" in pages
    assert "pages/11_edge_runtime_benchmark.py" in pages


def test_native_streamlit_theme_matches_product_shell() -> None:
    config = Path(".streamlit/config.toml").read_text(encoding="utf-8")
    assert 'backgroundColor = "#080B10"' in config
    assert "showSidebarNavigation = false" in config
