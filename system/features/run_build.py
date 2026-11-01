"""
System Features — CLI entry point.

Usage:
    python -m system.features.run_build --start 2017-01-01 --end 2026-10-06
    python -m system.features.run_build --dates 2024-07-01 2024-07-02
    python -m system.features.run_build --no-partition
"""
from system.features.build import main

if __name__ == "__main__":
    main()
