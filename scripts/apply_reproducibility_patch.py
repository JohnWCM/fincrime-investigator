"""Patch the EXISTING checked-out Streamlit app without replacing user edits.

Run after extracting the reproducibility ZIP to the repository root:
  python -m scripts.apply_reproducibility_patch

The process validates the patched Python before writing. It never changes Git
history, data/processed, user secrets, or tracked benchmark files.
"""
from __future__ import annotations

import ast
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ["app/data_config.py", "scripts/bootstrap_demo.py", "scripts/verify_reproduction.py"]


def patch_app(original: str) -> str:
    if "SUBSET, MANIFEST = resolve_data_paths(ROOT)" in original:
        return original
    import_anchor = "from app.ui_helpers import "
    if original.count(import_anchor) != 1:
        raise ValueError("Cannot find unique app.ui_helpers import; refusing unsafe edits")
    patched = original.replace(import_anchor,
                               "from app.data_config import resolve_data_paths, is_generated_demo\n" + import_anchor, 1)
    subset = re.compile(r"(?m)^SUBSET\s*=\s*ROOT\s*/\s*['\"][^'\"\n]+['\"]\s*$")
    manifest = re.compile(r"(?m)^MANIFEST\s*=\s*ROOT\s*/\s*['\"][^'\"\n]+['\"]\s*$")
    if len(subset.findall(patched)) != 1 or len(manifest.findall(patched)) != 1:
        raise ValueError("SUBSET/MANIFEST assignments differ from expected layout; refusing unsafe edits")
    patched = subset.sub("SUBSET, MANIFEST = resolve_data_paths(ROOT)", patched, count=1)
    patched = manifest.sub("", patched, count=1)

    anchor = 'def caption_note() -> None:\n'
    if patched.count(anchor) != 1:
        raise ValueError("caption_note signature differs; refusing unsafe edits")
    patched = patched.replace(anchor, anchor + (
        '    if is_generated_demo(SUBSET):\n'
        '        st.info("GENERATED DEMO DATASET: fictional example only, not IBM AMLworld. "\n'
        '                "The published benchmark scores and IBM case examples do not describe these records.")\n'
    ), 1)

    # Keep historical benchmark numbers visible but explicitly separate from demo.
    bm = 'def benchmark_lab() -> None:\n'
    if bm in patched:
        patched = patched.replace(bm, bm + (
            '    if is_generated_demo(SUBSET):\n'
            '        st.info("Saved benchmark summaries were produced on IBM AMLworld and FATF "\n'
            '                "evaluation questions; they have not been rerun on this generated demo dataset.")\n'
        ), 1)

    # Display the right provenance in the status caption, without changing
    # existing metrics, routes, widgets or investigation logic.
    original_caption = 'st.caption("Synthetic IBM AML transactions · Read-only analysis · Heuristics are not a determination of criminal conduct · No case labels exposed")'
    if original_caption in patched:
        patched = patched.replace(original_caption,
            'st.caption(("Generated fictional demo" if is_generated_demo(SUBSET) else "Synthetic IBM AML transactions") '
            '+ " · Read-only analysis · Heuristics are not a determination of criminal conduct · No case labels exposed")')
    patched = patched.replace(
        'Run `python -m src.ingest` and `python -m src.prepare_data` in the project root.',
        'For a new clone run `python -m scripts.bootstrap_demo` in the project root. '
        'For IBM data, see docs/FRESH_CLONE.md.')
    ast.parse(patched)
    return patched


def patch_readme(original: str) -> str:
    if "## Fresh-clone demo: one reproducible setup" in original:
        return original
    anchor = "## Run locally"
    if original.count(anchor) != 1:
        raise ValueError("README missing expected Run locally header; refusing unsafe edits")
    new = '''## Fresh-clone demo: one reproducible setup

A fresh clone contains **no original IBM transaction rows** and intentionally does not
commit generated Parquet data, API credentials, FATF PDFs, or model caches. To run the
**complete application against a small, locally generated fictional demo** on Windows:

```powershell
py -3.11 -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-agent.txt
python -m scripts.bootstrap_demo
python -m scripts.verify_reproduction --strict-rag
python -m pytest -q
python -m streamlit run app/streamlit_app.py
```

The bootstrap script generates a deterministic, **non-IBM** dataset under
`data/demo/` using the repository's existing CSV-to-Parquet ingestion code,
and downloads the three publicly available FATF PDFs, extracts page chunks and
builds the embedding index. **Internet access is required** for the FATF PDFs,
Python packages and embedding model. If a FATF download fails, follow the
instructions in [the fresh-clone guide](docs/FRESH_CLONE.md), obtain the PDF
from its official publication page and rerun the command. The agent additionally
requires an API key and an enabled model; deterministic features do not.

The Streamlit app now automatically selects the first matching prepared dataset:
IBM two-hop, IBM expanded, IBM baseline, then the generated demo. To force the
sample dataset in PowerShell, run `$env:FINCRIME_DATASET = "demo"` before launching
Streamlit. No source-code path edits are necessary.

**Important:** Stored retrieval and SQL benchmark results refer to the original
IBM/FATF evaluation runs. They do **not** measure the synthetic quick-start
fixture. Exact reproduction of the published IBM figures additionally requires
the original `HI-Small_Trans.csv` and the evaluation procedures below.

'''
    updated = original.replace(anchor, new + anchor, 1)
    # Remove stale instructions requiring source-code edits after dataset selection
    # became environment-driven. Keep the IBM preparation instructions intact.
    old = 'The working application can use the selected baseline, expanded or two-hop dataset. In `app/streamlit_app.py`, make `SUBSET` and `MANIFEST` point to matching files. For the two-hop demonstration:\n\n```python\nSUBSET = ROOT / "data/processed/investigation_subset_2hop_demo.parquet"\nMANIFEST = ROOT / "data/processed/subset_manifest_2hop_demo.json"\n```\n'
    updated = updated.replace(old,
        'Dataset selection is automatic. To select an IBM dataset explicitly, set '
        '`$env:FINCRIME_DATASET = "twohop"` (or `expanded`, `ibm`, `demo`) before running Streamlit. '
        'Never edit paths in application source code.\n')
    return updated


def apply(root: Path = ROOT) -> list[str]:
    for rel in REQUIRED:
        if not (root / rel).is_file():
            raise FileNotFoundError(f"Missing {rel}; extract the ZIP at repository root first")
    app = root / "app/streamlit_app.py"
    readme = root / "README.md"
    ignore = root / ".gitignore"
    example = root / ".env.example"
    baseline = root / "src/baseline.py"
    existing = {p: p.read_text(encoding="utf-8") for p in (app, readme, ignore, example, baseline)}
    modified = dict(existing)
    modified[app] = patch_app(existing[app])
    modified[readme] = patch_readme(existing[readme])
    if "/data/demo/" not in existing[ignore]:
        modified[ignore] = existing[ignore].rstrip() + "\n\n# Generated fictional quick-start dataset (never commit local data/labels)\n/data/demo/\n"
    # The agent actually reads OPENAI_MODEL; preserve the rest of the user's template.
    modified[example] = re.sub(r"(?m)^FINCRIME_MODEL=", "OPENAI_MODEL=", existing[example])
    if "OPENAI_MODEL=" not in modified[example]:
        modified[example] += "\nOPENAI_MODEL=\n"
    modified[baseline] = existing[baseline].replace(
        'IBM data are synthetic; labels are withheld from this investigator.',
        'Transactions are synthetic (IBM AMLworld or locally generated fixture); labels are withheld from this investigator.')
    modified[baseline] = modified[baseline].replace(
        'Subset excludes edges outside selected induced networks.',
        'This analysis is limited to the selected prepared dataset; transactions outside it may be missing.')
    # Validate everything first; never partially edit when anchors do not match.
    ast.parse(modified[app])
    ast.parse(modified[baseline])
    changed = []
    for path, contents in modified.items():
        if contents != existing[path]:
            path.write_text(contents, encoding="utf-8", newline="\n")
            changed.append(str(path.relative_to(root)))
    return changed


if __name__ == "__main__":
    try:
        updated = apply()
        print("Updated: " + (", ".join(updated) if updated else "no files; already patched"))
    except Exception as e:
        raise SystemExit(f"Patch not applied: {type(e).__name__}: {e}") from e
