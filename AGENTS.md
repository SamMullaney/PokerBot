# AGENTS.md: Rules for AI assistants working on PokerBot

This file tells AI coding assistants (Claude, Codex, Copilot, Cursor, …) **how to work** in this repo. For **what the project is and how it works**, read [README.md](README.md) first.

---

## 1. Git workflow

**Never commit directly to `main`.** `main` must always run.

1. **Start from an up-to-date `main`** before any change:
   ```powershell
   git checkout main
   git pull
   ```
2. **Create one branch per change.** Keep each branch to one feature, fix or refactor, and don't mix unrelated changes.

   | Prefix | Use for | Example |
   |---|---|---|
   | `feature/` | New capability | `feature/pot-ocr` |
   | `fix/` | Bug fix | `fix/card-roi-bounds` |
   | `model/` | Training, data, or model-weight changes | `model/card-cnn-v5` |
   | `refactor/` | Restructuring with no behavior change | `refactor/roi-helpers` |
   | `chore/` | Docs, deps, config, tooling | `chore/update-readme` |

3. **Commit early and often** in small, logical steps. Don't save everything for one big commit at the end.
4. **Write clear commit messages:**
   - First line: imperative mood, 72 characters or fewer, says *what* changed (for example, `Add OCR for pot ROI`, not `stuff` or `fixed it`).
   - Blank line, then a body that explains *why* and anything that isn't obvious.
   - Leave unrelated files out of the commit. Check with `git status` / `git diff --staged` before committing.
5. **Merge when the work is done and tested** (see §2):
   ```powershell
   git checkout main
   git merge --no-ff <branch>
   git branch -d <branch>
   ```
   Only merge if `main` still runs afterwards. If the change is large or risky, stop and ask the user before merging.
6. **Never** force-push, rewrite published history, or push to the remote unless the user asks.
7. If you find uncommitted changes that aren't yours, **ask before you stash, commit or discard them.**

---

## 2. Testing: required before every merge

The repo has no automated test suite yet, so "tested" means **all** of these:

1. **Compile and import check:** must pass with no errors:
   ```powershell
   python -m compileall -q main.py config.py app capture state vision tools
   python -c "import config, main; from vision.card_detector import CardClassifier; CardClassifier(config.MODEL_CARD_PATH)"
   ```
2. **Unit tests for pure logic:** anything you add or change that runs without a screen goes in `tests/`, using `pytest`. That includes ROI math in `state/`, NMS and exclusion groups in `vision/player_detector.py`, parsing and encoding, and classifier output on saved images. Run:
   ```powershell
   python -m pytest
   ```
   Use small saved images in `tests/fixtures/` instead of live capture. Install test dependencies with `pip install -r requirements-dev.txt`.
3. **Live run, for any change to capture, vision, drawing or the main loop:** `python main.py` needs a poker table on screen. If you can't run it yourself, **say so explicitly** and ask the user to check it. Never claim it works if you haven't seen it work.
4. **Report results honestly.** Say what you ran, what passed and what failed. Don't hide or work around failures.

---

## 3. Code conventions

- **Configuration:** put tunable values (thresholds, paths, devices, FPS) in [config.py](config.py), not in module code. Machine-specific values should also be readable from the environment.
- **Secrets:** never hardcode keys, tokens or passwords. Put them in `.env` (git-ignored), read them with `os.getenv` in `config.py`, and add a placeholder line to [.env.example](.env.example). Never commit `.env`, and never print or log secret values.
- **No personal or absolute paths** such as `C:\Users\...`. Use paths relative to `config.BASE_DIR`, or take them as CLI arguments.
- **ROIs are table-relative** `(x_pct, y_pct, w_pct, h_pct)` fractions of the detected table box. Never use raw screen pixels. Convert with `TableBox.roi_from_rel()`.
- **Images are BGR** (OpenCV) throughout. Convert explicitly when a library expects RGB, as `CardClassifier` does.
- **Hot-loop performance:** `main.py` runs at screen FPS. Don't allocate models, windows or files inside the loop, and say so if a change noticeably affects FPS.
- **Structure:** keep the existing layout (`capture/`, `vision/`, `state/`, `app/`, `tools/`). Dev utilities go in `tools/` and run with `python -m tools.<name>` from the project root.
- **Dependencies:** add or remove packages in [requirements.txt](requirements.txt) with pinned versions, in the same commit as the code that needs them.
- **Docs:** if a change affects setup, configuration, the pipeline or the tools, update [README.md](README.md) in the same branch.

---

## 4. Models and data

- Active weights live in `vision/models/` and are referenced **only through `config.py`**: `MODEL_TABLE_PATH`, `MODEL_CARD_PATH`.
- When you train a new model version, give it a new versioned filename (for example, `..._cardv5.pt`). Point the config at it only after it beats the current model, then remove the old file in a separate commit (it stays in git history).
- **Don't delete `vision/models/52cards/`.** Those are the card reference images.
- Training crops and captured screenshots go in `data/`, which is git-ignored. Don't commit bulk training data.

---

## 5. Current state and known issues

- **Card classifier accuracy is poor** even though the card ROIs line up correctly. More, and more standardized, training data is the next step. This is the current priority.
- The card debug panel can overlap the main view at some window sizes.
- OCR, dealer button detection, state encoding and the decision engine are not built yet (see README Roadmap).

Update this section when an issue is fixed or a new one is found.

---

## 6. When in doubt

Ask the user before you:
- delete files, models or data
- change ROI geometry in `config.py`; it's calibrated by hand against the client
- add a large dependency
- merge something you couldn't fully test
