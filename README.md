# AHG_OCR_Pipeline_v2

A modular, real‑time OCR engine built around a **three‑model ensemble** (EAST, Tesseract, EasyOCR) with a reliability‑scored pipeline, asynchronous processing, a custom HUD overlay, and optional voice integration.  
Originally designed for accessibility use cases, the system has evolved into a general‑purpose **real‑time perception stack** suitable for AR overlays, automation, screen reading, and research.

---

## ✨ Key Features

- **Three‑Model OCR Ensemble**
  - **EAST** for text detection
  - **Tesseract** for classical OCR with dictionary bias
  - **EasyOCR** for deep‑learning recognition
  - Combined using a reliability‑scored arbitration system

- **Two Complete Pipelines**
  - **Normal Pipeline** — multi‑phase, recursive, reliability‑driven
  - **Async Pipeline** — threaded, mode‑timed, real‑time screen capture

- **Custom HUD Overlay**
  - Real‑time subtitles and text overlays
  - Themeable rendering layer
  - Designed for AR‑style augmentation

- **Voice Subsystem (Optional)**
  - Vosk‑based speech recognition
  - Subtitle engine + punctuation
  - Fully asynchronous

- **Server Layer**
  - HTTP server
  - Camera streaming
  - OCR task orchestration
  - UI templates

- **Extensive Testing Suite**
  - Benchmark images (clear, complex, scene, dummy)
  - PDF samples
  - Live OCR runners
  - Diagnostics + summary tables

---

## 🧩 Project Structure

```
app/                Application entrypoints (camera, server, config)
graphics/           HUD overlay, renderer, themes
ocr_modules/        OCR engines, pipelines, preprocessing, scoring
    base_modules/   EAST boxes, preprocess, parsers, reliability
    pipeline_utils/ Async pipeline, modes, phases, race logic
resources/          Models (EAST, Vosk), corpus, frequencies
server_utils/       HTTP server, camera, stream loop, UI templates
shared/             Diagnostics, frame buffers, summaries, helpers
testing/            Images, PDFs, runners, diagnostics, benchmarks
voice/              Async voice engine, recognizer, subtitles
```

---

## 🔍 Normal OCR Pipeline (Multi‑Model, Reliability‑Driven)

The standard pipeline uses all three OCR engines with a multi‑phase flow:

```
EAST + Tesseract
        ↓
EasyOCR + EAST output
        ↓
EasyOCR recursion (if unreliable)
        ↓
Text Output
        ↓
If still unreliable → No Text Output
```

### Reliability Logic
- EAST provides bounding boxes  
- Tesseract provides structured text  
- EasyOCR provides deep‑learning recognition  
- A scoring system determines:
  - **IsReliable** → accept output  
  - **IsNotReliable** → recurse or fail  
- Recursion is bounded by **mode‑based time limits**

Modes:
- `fast` — minimal recursion, low latency  
- `steady` — balanced  
- `extended` — maximum reliability  

---

## ⚡ Async Pipeline (Real‑Time, Threaded)

The async engine is designed for **real‑time screen capture** and runs independently of the main thread.

### Core Behavior
- Converts cv2 → PIL  
- Enforces mode‑based timing (`min_interval`)  
- Uses a `ThreadPoolExecutor`  
- Dispatches frames to `AsyncPipeline`  
- Calls a callback with results  
- Never blocks the main loop  

### Example (simplified)
```python
engine = AsyncOCREngine(mode="steady")

def on_result(result):
    print(result["text"])

engine.process(frame, callback=on_result)
```

This architecture is suitable for:
- AR glasses  
- HUD overlays  
- Desktop screen readers  
- Real‑time automation  
- Continuous monitoring systems  

---

## 🎨 HUD Overlay System

Located in `graphics/`:

- `renderer.py` — draws bounding boxes, subtitles, highlights  
- `overlay.py` — manages layers and blending  
- `theme.py` — colors, fonts, styles  

Designed for:
- real‑time subtitles  
- AR‑style augmentation  
- screen overlays  
- live diagnostics  

---

## 🔊 Voice Subsystem (Optional)

Located in `voice/`:

- Vosk‑based speech recognition  
- Async voice engine  
- Subtitle engine  
- Punctuation + cleanup  

Integrates with the HUD for:
- live captions  
- voice‑driven OCR modes  
- accessibility workflows  

---

## 🧪 Testing & Benchmarking

The `testing/` directory includes:

- **Benchmark_Images** (clear, complex, scene, dummy)
- **PDF samples**
- **Live OCR runners**
- **Diagnostics outputs**
- **Pipeline summaries**
- **OCR race comparisons**
- **Voice tests**

This makes the project suitable for:
- research  
- benchmarking  
- regression testing  
- model comparison  

---

## 🚀 Installation

```
pip install -r requirements.txt
```

Models are included in `resources/`:
- `east_model.pb`
- `vosk_model_small/`

---

## ▶️ Usage

### Run the main application
```
python app/main.py
```

### Run the camera OCR
```
python app/camera_runner.py
```

### Run the server
```
python app/server_runner.py
```

---

## 🧠 Why This Project Matters

AHG_OCR_Pipeline_v2 is more than an OCR script.  
It’s a **modular perception engine** built around:

- multi‑model fusion  
- reliability scoring  
- asynchronous processing  
- real‑time overlays  
- voice integration  
- a complete testing suite  

It can serve as:
- an accessibility tool  
- an AR overlay engine  
- a research platform  
- a real‑time automation module  
- a subsystem of a larger device (e.g., AHGadget)  

---

## 📜 License

MIT License



