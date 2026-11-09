"""Automated verification script ensuring 100% of DOM IDs accessed by frontend/app.js

exist in frontend/index.html.
"""

from __future__ import annotations

import re
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    js_path = root / "frontend" / "app.js"
    html_path = root / "frontend" / "index.html"

    js_content = js_path.read_text(encoding="utf-8")
    html_content = html_path.read_text(encoding="utf-8")

    # Extract all getElementById matches from app.js
    js_ids = set(re.findall(r'getElementById\(["\']([^"\']+)["\']\)', js_content))

    # Extract all id="..." attributes from index.html
    html_ids = set(re.findall(r'id=["\']([^"\']+)["\']', html_content))

    print(f"Total getElementById calls in app.js: {len(js_ids)}")
    print(f"Total id attributes in index.html: {len(html_ids)}")

    missing = sorted(list(js_ids - html_ids))
    if missing:
        print(f"FAILED: The following IDs from app.js are missing in index.html: {missing}")
        raise SystemExit(1)

    print("SUCCESS: 100% of getElementById IDs in app.js exist in index.html!")
    for idx, element_id in enumerate(sorted(js_ids), 1):
        print(f"  {idx:02d}. #{element_id} -> Present in index.html")


if __name__ == "__main__":
    main()

