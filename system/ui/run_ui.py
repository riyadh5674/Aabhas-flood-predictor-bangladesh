"""
System UI — CLI entry point.

Usage:
    python system/ui/run_ui.py              # start Streamlit on port 8501
    python system/ui/run_ui.py --port 8502  # custom port
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit.cli  # noqa: E402


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Run the Flood Predictor UI")
    ap.add_argument(
        "--port", type=int, default=8501, help="Streamlit server port"
    )
    ap.add_argument(
        "--host", default="localhost", help="Streamlit server host"
    )
    args = ap.parse_args()

    sys.argv = [
        "streamlit",
        "run",
        str(Path(__file__).resolve().parent / "app.py"),
        "--server.port", str(args.port),
        "--server.address", args.host,
    ]
    streamlit.cli.main()


if __name__ == "__main__":
    main()
