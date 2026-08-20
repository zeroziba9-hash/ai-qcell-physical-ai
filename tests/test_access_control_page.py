from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_access_control_page_issues_authenticated_session(monkeypatch) -> None:
    monkeypatch.setenv("QCELL_AUTH_MODE", "demo")
    page = Path(__file__).resolve().parents[1] / "pages" / "13_access_control.py"
    app = AppTest.from_file(page, default_timeout=90).run()

    assert not app.exception
    assert [field.label for field in app.text_input] == ["사용자 이름", "비밀번호"]
    app.text_input[0].set_value("quality")
    app.text_input[1].set_value("qcell-quality")
    app.button[0].click().run()

    assert not app.exception
    assert [button.label for button in app.button] == ["로그아웃"]
    principal = app.session_state.filtered_state["qcell_principal"]
    assert principal["username"] == "quality"
    assert principal["role"] == "quality_manager"
