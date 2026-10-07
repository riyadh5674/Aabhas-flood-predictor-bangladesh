import re
import sys

import requests

sys.stdout.reconfigure(encoding="utf-8")
H = {"User-Agent": "Mozilla/5.0"}
for u in [
    "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/",
    "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/tifs/",
    "https://data.chc.ucsb.edu/products/CHIRPS-2.0/bdaily/tifs/",
]:
    try:
        r = requests.get(u, headers=H, timeout=60)
        print(u, r.status_code, len(r.text))
        print(re.findall(r'href="([^"]+)"', r.text)[:30])
    except Exception as e:
        print(u, "ERR", e)
