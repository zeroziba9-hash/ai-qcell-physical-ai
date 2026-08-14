"""Container health probe that follows the configured service port."""

from __future__ import annotations

import os
import urllib.request


def health_url(environment: dict[str, str] | None = None) -> str:
    values = os.environ if environment is None else environment
    port = int(values.get("PORT", "8501"))
    return f"http://127.0.0.1:{port}/_stcore/health"


def main() -> None:
    with urllib.request.urlopen(health_url(), timeout=4) as response:
        if response.status >= 400:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
