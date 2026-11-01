"""
System Map — CLI entry point.

Usage:
    python system/map/run_map.py --date 2024-07-15 --horizon 24h --level adm2
    python system/map/run_map.py --date 2024-07-15 --all-horizons --all-levels
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from system.map.choropleth import main

if __name__ == "__main__":
    main()
