"""Safely install Presentation Mode into an existing FinCrime Investigator app.

Run once from the repository root after extracting the ZIP:
  python -m scripts.install_presentation_mode

Only app/streamlit_app.py is edited. The original is saved to a local .bak file.
This patch intentionally does not change any investigation logic or datasets.
"""
from __future__ import annotations

import ast
from pathlib import Path
import re
import shutil


def patch_app(source: str) -> str:
    if 'render_presentation_mode(ROOT, cases)' in source:
        return source

    if 'from app.presentation_mode import render_presentation_mode' in source:
        raise RuntimeError('Partial Presentation Mode integration found; inspect the app manually.')

    import_anchor = 'from app.ui_helpers import '
    if source.count(import_anchor) != 1:
        raise RuntimeError('Cannot locate unique app.ui_helpers import.')
    updated = source.replace(import_anchor,
        'from app.presentation_mode import render_presentation_mode\n' + import_anchor, 1)

    nav_pattern = re.compile(r'(?m)^([ \t]*view\s*=\s*st\.radio\(["\']Workspace["\'],\s*)(\[[^\]\n]*\])([^\n]*\))')
    matches = list(nav_pattern.finditer(updated))
    if len(matches) != 1:
        raise RuntimeError('Cannot safely locate the sidebar Workspace radio.')
    m = matches[0]
    values = ast.literal_eval(m.group(2))
    if not isinstance(values, list) or 'Investigation' not in values:
        raise RuntimeError('The Workspace navigation options differ from expected layout.')
    if 'Presentation Mode' not in values:
        values.insert(0, 'Presentation Mode')
    tail = m.group(3)
    if 'key=' not in tail:
        tail = tail[:-1] + ', key="workspace_view")'
    else:
        if 'key="workspace_view"' not in tail and "key='workspace_view'" not in tail:
            raise RuntimeError('Workspace radio already has another key; please integrate manually.')
    repl = m.group(1) + repr(values) + tail
    updated = updated[:m.start()] + repl + updated[m.end():]

    case_anchor = 'st.selectbox("Case account", cases, disabled=not cases)'
    if case_anchor not in updated:
        if 'key="case_account"' not in updated:
            raise RuntimeError('Cannot safely locate sidebar Case account selectbox.')
    else:
        if updated.count(case_anchor) != 1:
            raise RuntimeError('Ambiguous Case account selectbox.')
        updated = updated.replace(case_anchor,
            'st.selectbox("Case account", cases, disabled=not cases, key="case_account")', 1)

    title_anchor = 'st.title("FinCrime Investigator")\ncaption_note()'
    if updated.count(title_anchor) == 1:
        updated = updated.replace(title_anchor,
            'if view != "Presentation Mode":\n    st.title("FinCrime Investigator")\n    caption_note()', 1)
    elif 'if view != "Presentation Mode":' not in updated:
        raise RuntimeError('Cannot safely locate main title/caption block.')

    dispatch = 'if view == "Investigation":\n    investigation_workspace(account)'
    if updated.count(dispatch) != 1:
        raise RuntimeError('Cannot safely locate top-level Investigation dispatcher.')
    updated = updated.replace(dispatch,
        'if view == "Presentation Mode":\n'
        '    render_presentation_mode(ROOT, cases)\n'
        'elif view == "Investigation":\n'
        '    investigation_workspace(account)', 1)

    ast.parse(updated, filename='app/streamlit_app.py')
    return updated


def install(root: Path) -> str:
    app = root / 'app' / 'streamlit_app.py'
    presentation = root / 'app' / 'presentation_mode.py'
    if not app.is_file() or not presentation.is_file():
        raise FileNotFoundError('Expected app/streamlit_app.py and app/presentation_mode.py. Extract ZIP into repo root.')
    source = app.read_text(encoding='utf-8')
    updated = patch_app(source)
    if updated == source:
        return 'Already installed; no changes.'
    backup = app.with_name('streamlit_app.before_presentation_mode.py.bak')
    if backup.exists():
        raise FileExistsError(f'Backup already exists, refusing to overwrite: {backup}')
    shutil.copy2(app, backup)
    app.write_text(updated, encoding='utf-8', newline='\n')
    return 'Presentation Mode installed. Original app saved to ' + str(backup)


if __name__ == '__main__':
    try:
        print(install(Path.cwd()))
    except Exception as exc:
        raise SystemExit(f'INSTALL FAILED (no files changed): {type(exc).__name__}: {exc}') from exc
