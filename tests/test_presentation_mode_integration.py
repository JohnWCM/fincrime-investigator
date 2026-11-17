"""Presentation Mode installer and slide evidence smoke tests (offline)."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "presentation_mode_installer_under_test", ROOT / "scripts" / "install_presentation_mode.py"
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def original_app() -> str:
    # Existing production app contains the following stable navigation anchors.
    return '''from app.ui_helpers import DISPLAY_NAMES
import streamlit as st
ROOT = "repo"
cases = ["example"]
account = st.selectbox("Case account", cases, disabled=not cases) if cases else None
view = st.radio("Workspace", ["Investigation", "Account Intelligence", "Network", "FATF Evidence", "Dashboard", "Benchmark Lab"])
st.title("FinCrime Investigator")
caption_note()
if view == "Investigation":
    investigation_workspace(account)
elif view == "Account Intelligence":
    account_intelligence_view(account, SUBSET)
elif view == "Network":
    network_explorer(account)
'''


def test_installer_adds_slide_menu_and_preserves_pages():
    patched = module.patch_app(original_app())
    ast.parse(patched)
    assert 'render_presentation_mode(ROOT, cases)' in patched
    assert 'key="workspace_view"' in patched
    assert 'key="case_account"' in patched
    assert 'elif view == "Account Intelligence"' in patched
    assert 'elif view == "Network"' in patched
    assert patched.index("Presentation Mode") < patched.index("Investigation", patched.index("st.radio"))


def test_installer_idempotent():
    once = module.patch_app(original_app())
    assert module.patch_app(once) == once


def test_installer_fails_without_expected_navigation():
    broken = original_app().replace('st.radio("Workspace"', 'st.radio("Another menu"')
    try:
        module.patch_app(broken)
        assert False, "Installer should refuse unexpected menu"
    except RuntimeError:
        pass


def test_slide_module_compiles_without_streamlit_installation():
    path = ROOT / "app" / "presentation_mode.py"
    ast.parse(path.read_text(encoding="utf-8"))


def test_integration_preserves_data_guard():
    source = original_app().replace('st.title("FinCrime Investigator")',
        'if view in ("Investigation", "Network"):\n    requirement_guard()\nst.title("FinCrime Investigator")')
    updated = module.patch_app(source)
    assert 'if view in ("Investigation", "Network"):' in updated
    assert 'requirement_guard()' in updated
