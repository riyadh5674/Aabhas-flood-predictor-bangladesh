"""
System Alerts — CLI entry point.

Usage:
    python system/alerts/run_alerts.py --date 2024-07-15 --horizon 24h
    python system/alerts/run_alerts.py --date 2024-07-15 --all-horizons
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from system.alerts.alert_engine import main

if __name__ == "__main__":
    main()
