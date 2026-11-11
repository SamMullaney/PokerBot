# PokerBot

A real-time computer-vision pipeline that watches an online poker client on screen and reads the table state: where the tables are, which seats are occupied, and the player's hole cards and the community cards.

The long-term goal is to feed this state into a GTO decision engine (see [Roadmap](#roadmap)). Right now the project covers the **perception** layer and shows its output in a live debug view.

> **Platform:** Windows only. Screen capture uses [DXcam](https://github.com/ra1nty/DXcam), which relies on the Windows Desktop Duplication API.

---

## How it works

Every captured frame goes through this pipeline:

```
DXcam screen frame (BGR)
   │
   ▼
TableDetector ── YOLO model (vision/models/table_detector_v1.pt)
   │              → one TableBox (pixel bbox + confidence) per table
   ▼
For each table:
   ├── PlayerDetector ─ classic CV, no model
   │      • crops each seat's "occupancy" ROI
   │      • occupied if Canny edge density AND Laplacian variance both pass thresholds
   │      • resolves overlapping seat layouts (SEAT_EXCLUSION_GROUPS + NMS)
   │      • occupied seats also get name / VPIP / stack / bet / position ROIs
   │
   ├── CardClassifier ─ TinyCornerNet CNN (vision/models/tiny_corner_net_best_cardv4.pt)
   │      • crops the 2 player-card and 5 community-card ROIs
   │      • predicts rank (13 classes) and suit (4 classes) from the card corner
   │      • card_conf = rank_conf × suit_conf; below CARD_CONF_THRES → NO_CARD
   │
   ▼
DebugViewer ── annotated frame + "Debug Output" panel listing cards and FPS
```

### Table-relative ROIs

Every region of interest in [config.py](config.py) is stored as a **fraction of the detected table box** `(x_pct, y_pct, w_pct, h_pct)`, not as fixed pixels. `TableBox.roi_from_rel()` converts them to screen pixels for each frame. Because of this, the ROIs keep working when the table window moves or is resized, as long as the client's layout stays the same.

- `TABLE_ROIS`: card slots and pot area
- `SEAT_ROIS`: per-seat regions for every seat layout the client uses (6-max, 9-max, …). Seats that occupy the same physical spot in different layouts are listed in `SEAT_EXCLUSION_GROUPS`, so only one of them is marked occupied.

---

## Project structure

```
PokerBot/
├── main.py                  # Entry point: capture → detect → classify → display loop
├── config.py                # All settings: env overrides, thresholds, ROI geometry
├── requirements.txt
├── .env.example             # Template for local settings (copy to .env)
│
├── capture/
│   └── screen_capture.py    # DXcam wrapper (ScreenCapture)
├── vision/
│   ├── table_detector.py    # YOLO table detection → TableBox list
│   ├── player_detector.py   # Seat occupancy via edge/texture metrics + NMS
│   ├── card_detector.py     # TinyCornerNet + CardClassifier
│   ├── draw.py              # Drawing helpers for the debug overlay
│   └── models/              # Trained weights (.pt) and 52-card reference images
├── state/
│   └── table_state.py       # TableBox dataclass + relative-ROI math
├── app/
│   └── debug_viewer.py      # OpenCV windows, card panel, FPS logging
└── tools/                   # Development utilities (run as modules, see below)
    ├── capture_card_rois.py      # Save card crops for CNN training data
    ├── measure_table_bounds.py   # Click to get a table's pixel bbox on a screenshot
    ├── calibrate_rois.py         # Click to generate table-relative ROI tuples
    └── test_table_detection.py   # Live YOLO-only sanity check
```

---

## Setup

Requires **Python 3.10+** (developed on 3.13) on Windows.

```powershell
git clone https://github.com/SamMullaney/PokerBot.git
cd PokerBot

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt

copy .env.example .env      # optional: edit to change device, thresholds, model paths
```

**GPU (optional):** `requirements.txt` installs the CPU build of PyTorch. To run the table detector on an NVIDIA GPU, first install the CUDA build of torch (the exact command is in a comment at the top of `requirements.txt`), then set `DEVICE=cuda` in `.env`.

---

## Running

Open the poker client with a table visible, then run this from the project root:

```powershell
python main.py
```

Two windows open:

| Window | Shows |
|---|---|
| **PokerBot Debug (q to quit)** | The screen with table boxes (green), seat ROIs (orange), card ROIs (blue) and card labels drawn on top |
| **Debug Output** | Detected player and community cards with confidences, plus recent FPS readings |

Focus either window and press **`q`** to quit.

> **Windows tip:** if `python` opens the wrong interpreter (for example, the Microsoft Store `python3`), activate the venv first or call `.\.venv\Scripts\python.exe main.py` directly.

---

## Configuration

Machine-specific settings come from `.env`. Every key is optional, and `.env.example` lists the defaults.

| Variable | Default | Purpose |
|---|---|---|
| `DEVICE` | `cpu` | Device for the YOLO table detector (`cpu`, `cuda`, `cuda:0`) |
| `CARD_DEVICE` | `cpu` | Device for the card CNN |
| `MODEL_TABLE_PATH` | `vision/models/table_detector_v1.pt` | Table detector weights |
| `MODEL_CARD_PATH` | `vision/models/tiny_corner_net_best_cardv4.pt` | Card classifier weights |
| `TABLE_CONF_THRES` | `0.7` | Minimum YOLO confidence for a table |
| `CARD_CONF_THRES` | `0.6` | Minimum rank × suit confidence before a card is labeled `NO_CARD` |
| `CAPTURE_FPS` | `30` | Target screen-capture frame rate |
| `CAPTURED_CARDS_DIR` | `data/captured_cards` | Output folder for `tools/capture_card_rois.py` |

Layout geometry and player-detector tuning (`TABLE_ROIS`, `SEAT_ROIS`, `EDGE_RATIO_THRESHOLD`, `LAPLACIAN_VAR_THRESHOLD`, `NMS_OVERLAP_THRESHOLD`) is set directly in [config.py](config.py), because it depends on the poker client rather than the machine.

**Secrets:** the project has no API keys yet. If you add any later, put them in `.env` (which is git-ignored), read them in `config.py` with `os.getenv`, and add a placeholder line to `.env.example`.

---

## Developer tools

Run each tool as a module from the project root so the project imports resolve:

```powershell
# Collect training crops: press c to capture, q to quit
python -m tools.capture_card_rois

# 1) Find the table's pixel bounds on a screenshot (click top-left, then bottom-right)
python -m tools.measure_table_bounds path\to\screenshot.png

# 2) Click ROI corners to get table-relative tuples to paste into config.py
python -m tools.calibrate_rois path\to\screenshot.png --table 565 235 1416 868

# Live check of the YOLO table detector by itself
python -m tools.test_table_detection
```

**Recalibrating for a new client or layout:** take a screenshot, run `measure_table_bounds` to get the table box, run `calibrate_rois` with that box, and paste the printed tuples into `TABLE_ROIS` / `SEAT_ROIS`.

---

## Roadmap

- [ ] OCR for pot, stack and bet regions
- [ ] Dealer button / blind detection (rule-based)
- [ ] Encode a structured game state (`state/`)
- [ ] Decision engine and action output
- [ ] More standardized card training data, and slightly larger card ROIs

### GTO notes

Candidate solver algorithms:

- **Fictitious Play**
- **Counterfactual Regret Minimization (CFR)**

Solvers only handle a limited set of bet sizes. Each extra bet-size interval makes the game tree much larger, until the computation becomes too slow or impossible. Memory needs are also high, so the solver will probably have to run in the cloud, and its core may need to be written in a faster language than Python.
