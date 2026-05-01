"""Build product_area taxonomy from the data/ folder tree.

The taxonomy is derived entirely from the corpus directory structure,
not hardcoded. This means any update to data/ automatically propagates.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Set


def build_taxonomy(data_dir: Path) -> Dict[str, Set[str]]:
    """Walk data/<company>/... and return {company -> set of area_paths}.

    area_path is the relative folder under data/<company>/, e.g.
    'settings/user-account-settings-and-preferences'.
    """
    taxonomy: Dict[str, Set[str]] = {}

    for company_dir in sorted(data_dir.iterdir()):
        if not company_dir.is_dir():
            continue
        company = company_dir.name
        areas: Set[str] = set()

        for md_file in company_dir.rglob("*.md"):
            rel = md_file.relative_to(company_dir)
            if len(rel.parts) > 1:
                area = "/".join(rel.parts[:-1])
            else:
                area = "_root"
            areas.add(area)

        taxonomy[company] = areas

    return taxonomy


def save_taxonomy(taxonomy: Dict[str, Set[str]], out_path: Path) -> None:
    serializable = {k: sorted(v) for k, v in taxonomy.items()}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def load_taxonomy(path: Path) -> Dict[str, Set[str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {k: set(v) for k, v in raw.items()}


def area_from_chunk_path(source_path: str, data_dir: Path) -> str:
    """Derive product_area from a chunk's source_path.

    source_path is stored relative to repo root, e.g.:
      'data/hackerrank/settings/user-account-settings/foo.md'
    We extract the first folder level under data/<company>/ as the area.
    This matches the expected label granularity (e.g. 'screen', 'community').
    """
    try:
        parts = Path(source_path).parts
        if "data" in parts:
            data_idx = parts.index("data")
            after_data = parts[data_idx + 1:]
            # after_data: (company, area1, ..., file.md)
            if len(after_data) <= 2:
                return after_data[0] if after_data else "general"
            # Return just the first-level area folder
            return after_data[1]
        return "general"
    except (ValueError, IndexError):
        return "general"
