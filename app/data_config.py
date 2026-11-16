"""Resolve data paths for fresh clones without modifying source code.

Existing IBM datasets always win over the generated fixture unless explicitly
selected via FINCRIME_DATASET=demo. No label files enter this module.
"""
from __future__ import annotations

import os
from pathlib import Path


def resolve_data_paths(root: Path, environ: dict[str, str] | None = None) -> tuple[Path, Path]:
    root = Path(root).resolve()
    env = os.environ if environ is None else environ
    explicit_subset = env.get("FINCRIME_SUBSET_PATH", "").strip()
    explicit_manifest = env.get("FINCRIME_MANIFEST_PATH", "").strip()
    if bool(explicit_subset) != bool(explicit_manifest):
        raise ValueError("Set FINCRIME_SUBSET_PATH and FINCRIME_MANIFEST_PATH together")

    def path(value: str) -> Path:
        candidate = Path(value).expanduser()
        return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()

    if explicit_subset:
        return path(explicit_subset), path(explicit_manifest)

    candidates = {
        "twohop": ("data/processed/investigation_subset_2hop_demo.parquet",
                   "data/processed/subset_manifest_2hop_demo.json"),
        "expanded": ("data/processed/investigation_subset_expanded.parquet",
                     "data/processed/subset_manifest_expanded.json"),
        "ibm": ("data/processed/investigation_subset.parquet",
                "data/processed/subset_manifest.json"),
        "demo": ("data/demo/processed/investigation_subset.parquet",
                 "data/demo/processed/subset_manifest.json"),
    }
    requested = env.get("FINCRIME_DATASET", "auto").lower().strip()
    if requested not in {"auto", *candidates}:
        raise ValueError(f"FINCRIME_DATASET must be auto, {', '.join(candidates)}")
    order = list(candidates) if requested == "auto" else [requested]
    for key in order:
        subset, manifest = (root / v for v in candidates[key])
        if subset.is_file() and manifest.is_file():
            return subset, manifest
    if requested != "auto":
        raise FileNotFoundError(f"Requested {requested!r} data was not prepared. See docs/FRESH_CLONE.md")
    # Keep a coherent fallback so Streamlit's existing requirement_guard can explain missing data.
    subset, manifest = candidates["demo"]
    return root / subset, root / manifest


def is_generated_demo(path: Path) -> bool:
    return "data/demo/processed/" in Path(path).as_posix().replace("\\", "/")
