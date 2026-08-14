import pytest

from scripts.container_healthcheck import health_url
from scripts.start_container import streamlit_command


def test_streamlit_command_uses_platform_port() -> None:
    assert streamlit_command({"PORT": "10000"}) == [
        "streamlit",
        "run",
        "app.py",
        "--server.address=0.0.0.0",
        "--server.port=10000",
    ]
    assert health_url({"PORT": "10000"}) == "http://127.0.0.1:10000/_stcore/health"


@pytest.mark.parametrize("port", ["abc", "0", "70000"])
def test_streamlit_command_rejects_invalid_port(port: str) -> None:
    with pytest.raises(ValueError, match="PORT"):
        streamlit_command({"PORT": port})
