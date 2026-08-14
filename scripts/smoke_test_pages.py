"""Run every Streamlit entry point and report uncaught rendering errors."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    scripts = [ROOT / "app.py", *sorted((ROOT / "pages").glob("*.py"))]
    failures: list[str] = []

    for script in scripts:
        relative = script.relative_to(ROOT).as_posix()
        app = AppTest.from_file(str(script), default_timeout=90).run()
        exceptions = list(app.exception)
        if exceptions:
            failures.append(f"{relative}: {exceptions[0].message}")
            print(f"FAIL  {relative}")
        else:
            print(f"OK    {relative}")

    if failures:
        raise SystemExit("\n".join(failures))

    print(f"\n{len(scripts)} Streamlit pages rendered without uncaught exceptions.")


if __name__ == "__main__":
    main()
