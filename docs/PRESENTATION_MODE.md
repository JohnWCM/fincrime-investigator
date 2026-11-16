# Presentation Mode (FinCrime Investigator)

Presentation Mode is a **guided, nine-screen presentation within Streamlit**. It does not overwrite or replicate the investigation workspaces.

## Install in the existing local project (Windows PowerShell)

1. From the repository root, confirm `git status --short` is clean. Make a separate Git branch if desired: `git switch -c feature/presentation-mode`.
2. Extract `fincrime-presentation-mode.zip` into that directory, preserving `app/`, `scripts/`, `tests/` and `docs/`.
3. Run `python -m scripts.install_presentation_mode`.
4. Run `python -m pytest -q`.
5. Start the application with `python -m streamlit run app/streamlit_app.py --server.port 8502`.
6. Choose **Presentation Mode** in the Streamlit sidebar and use **Next / Previous**.

The installer makes a local backup `app/streamlit_app.before_presentation_mode.py.bak` before editing, refuses unexpected navigation layouts, validates the result using Python's AST parser, and does nothing on subsequent runs. The backup should **not** be committed.

## Features

- Nine guided screens: title, problem, product, architecture, retrieval design, evaluation, engineering decisions, limitations and live-demo launch.
- Reported retrieval values are **read from the saved development JSON metrics** when available; the page never calculates fresh retrieval metrics during the talk.
- Existing screenshot files under `docs/screenshots/` are shown where relevant.
- Live demo buttons jump to the **existing Streamlit navigation** and preselect the intended account via a callback (avoids Streamlit widget-session-state modification errors).
- If the account does not exist in the currently selected dataset, its launch button is disabled rather than showing incorrect account data.
- Slide content works without a local Parquet dataset, an embedding index or an API key. The real investigation views still require their normal data and model setup.

## Recommended live-demo sequence

1. `023::800167090` — Account Intelligence: ten identical outgoing payments, counterparty concentration (when using the original prepared IBM data).
2. `00701::800A8E530` — Network: switch 1 hop to 2 hops (5 to 38 nodes for the prepared two-hop case).
3. `021575::80491D7A0` — Investigation: show the previously verified agent report, citations, and tool trace.

Launch accounts are only offered when they exist in the selected local manifest. Generated fictional data from the reproducibility patch may contain different account IDs, in which case Presentation Mode provides an available-account fallback.

The benchmark results describe specific historical IBM/FATF evaluations, **not** the generated fictional quick-start dataset and **not** general money-laundering detection accuracy.

## Troubleshooting

- If you see `ModuleNotFoundError: No module named 'app.presentation_mode'`, check that the ZIP was extracted at the repository root (not into an extra nested folder).
- If installation fails with `Cannot safely locate`, the current app's navigation differs from the version used to build the patch. **Do not paste arbitrary replacements.** Share the current sidebar and bottom dispatcher code for a targeted adjustment.
- If launch buttons are disabled, verify you selected the original IBM two-hop/expanded dataset and its matching manifest. Present slides without data if necessary.
- In the browser, use Streamlit's settings to choose a dark theme if you want the closest match to the deck.

## Commit after verification

```powershell
git add app/presentation_mode.py app/streamlit_app.py scripts/install_presentation_mode.py tests/test_presentation_mode_integration.py docs/PRESENTATION_MODE.md
git diff --cached --check
git commit -m "Add Streamlit presentation mode"
git push origin main
```
