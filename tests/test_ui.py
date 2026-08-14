from qcell.ui import GLOBAL_CSS, page_header_html


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
