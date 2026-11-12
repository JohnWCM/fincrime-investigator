"""Add Account Intelligence to the existing Streamlit app without replacing it.

Run from the fincrime-investigator repository root after extracting the upgrade:
    python scripts/install_account_intelligence.py
"""
from __future__ import annotations

import ast
from pathlib import Path
import shutil


def apply(root: Path) -> str:
    path = root / "app" / "streamlit_app.py"
    dependencies = [root / "app" / "account_intelligence.py",
                    root / "app" / "intelligence_data.py"]
    missing = [str(p) for p in [path, *dependencies] if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing installation files: " + ", ".join(missing))
    original = path.read_text(encoding="utf-8")
    if "account_intelligence_view(account, SUBSET)" in original:
        return "Already installed. Nothing changed."

    source = original
    import_anchor = "from app.ui_helpers import "
    if source.count(import_anchor) != 1:
        raise RuntimeError("Cannot locate unique UI import. No changes made.")
    source = source.replace(import_anchor,
                            "from app.account_intelligence import account_intelligence_view\n" + import_anchor,
                            1)

    nav_old = '["Investigation", "Network", "FATF Evidence", "Dashboard", "Benchmark Lab"]'
    nav_new = '["Investigation", "Account Intelligence", "Network", "FATF Evidence", "Dashboard", "Benchmark Lab"]'
    if source.count(nav_old) != 1:
        raise RuntimeError("Sidebar navigation differs from expected layout. No changes made.")
    source = source.replace(nav_old, nav_new, 1)

    dispatch_anchor = 'elif view == "Network":\n    network_explorer(account)'
    dispatch_new = '''elif view == "Account Intelligence":
    if not account:
        st.warning("Choose an account from the sidebar.")
        st.stop()
    if not SUBSET.is_file():
        st.error("The selected transaction Parquet file is missing.")
        st.stop()
    account_intelligence_view(account, SUBSET)
elif view == "Network":
    network_explorer(account)'''
    if source.count(dispatch_anchor) != 1:
        raise RuntimeError("App page dispatch differs from expected layout. No changes made.")
    source = source.replace(dispatch_anchor, dispatch_new, 1)
    ast.parse(source, filename=str(path))

    backup = path.with_name("streamlit_app.before_account_intelligence.py.bak")
    if backup.exists():
        raise FileExistsError(f"Backup already exists; refusing to replace: {backup}")
    shutil.copy2(path, backup)
    path.write_text(source, encoding="utf-8")
    return f"Installed Account Intelligence. Original app saved to {backup}"


if __name__ == "__main__":
    try:
        print(apply(Path.cwd()))
    except Exception as exc:
        raise SystemExit(f"INSTALL FAILED (existing app unchanged): {type(exc).__name__}: {exc}") from exc
