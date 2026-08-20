"""Start Streamlit on the platform-provided port."""

from __future__ import annotations

import os


def streamlit_command(environment: dict[str, str] | None = None) -> list[str]:
    values = os.environ if environment is None else environment
    raw_port = values.get("PORT", "8501")
    try:
        port = int(raw_port)
    except ValueError as error:
        raise ValueError("PORT must be an integer") from error
    if not 1 <= port <= 65_535:
        raise ValueError("PORT must be between 1 and 65535")
    return [
        "streamlit",
        "run",
        "app.py",
        "--server.address=0.0.0.0",
        f"--server.port={port}",
    ]


def main() -> None:
    command = streamlit_command()
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
