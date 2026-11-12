"""Safe, idempotent installer for the Account Intelligence Pattern alerts tab.

Run from the fincrime-investigator repository root after extraction:
    python scripts/install_intelligence_alerts.py
"""
from __future__ import annotations

import ast
import shutil
from pathlib import Path


def install(root: Path) -> str:
    app = root / "app" / "account_intelligence.py"
    required = [app, root / "app" / "intelligence_data.py",
                root / "app" / "intelligence_alerts.py",
                root / "app" / "intelligence_alerts_view.py"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing files: " + ", ".join(missing))

    source = app.read_text(encoding="utf-8")
    if "from app.intelligence_alerts_view import alerts_tab" in source:
        if "alerts_tab(fin, account_id, start, end, selected_currency)" in source:
            return "Pattern alerts already installed. No changes made."
        raise RuntimeError("Partial alert install found: please inspect the app manually.")
    # Preserve existing application modifications: only three exact local edits.
    import_anchor = "from app.intelligence_data import ("
    tabs_anchor = '''tab_timeline, tab_cp, tab_self, tab_ledger = st.tabs([
            "Activity timeline", "Counterparties", "Self-transfers", "Transaction ledger"
        ])'''
    dispatch_anchor = '''        with tab_self:
            _self_tab(fin, account_id, start, end, selected_currency)'''
    for anchor in (import_anchor, tabs_anchor, dispatch_anchor):
        if source.count(anchor) != 1:
            raise RuntimeError(f"Could not locate unique insertion point: {anchor[:65]!r}. No files changed.")
    patched = source.replace(import_anchor,
                             "from app.intelligence_alerts_view import alerts_tab\n" + import_anchor, 1)
    patched = patched.replace(tabs_anchor, '''tab_timeline, tab_cp, tab_alerts, tab_self, tab_ledger = st.tabs([
            "Activity timeline", "Counterparties", "Pattern alerts", "Self-transfers", "Transaction ledger"
        ])''', 1)
    patched = patched.replace(dispatch_anchor, '''        with tab_alerts:
            alerts_tab(fin, account_id, start, end, selected_currency)
        with tab_self:
            _self_tab(fin, account_id, start, end, selected_currency)''', 1)
    ast.parse(patched, filename=str(app))

    backup = app.with_name("account_intelligence.before_pattern_alerts.py.bak")
    if backup.exists():
        raise FileExistsError(f"Backup exists; refusing to overwrite: {backup}")
    shutil.copy2(app, backup)
    app.write_text(patched, encoding="utf-8")
    return f"Pattern alerts installed; original saved to {backup}"


if __name__ == "__main__":
    try:
        print(install(Path.cwd()))
    except Exception as exc:
        raise SystemExit(f"INSTALL FAILED: {type(exc).__name__}: {exc}") from exc
