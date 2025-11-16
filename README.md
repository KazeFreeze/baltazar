# Omnilingual ASR Headless Adapter

A clean, industry-standard Python adapter for Meta's Omnilingual ASR supporting **1600+ languages** including English and Filipino/Tagalog.

This adapter automatically handles audio resampling and chunks long audio files (>30 seconds) for seamless transcription.

## Features

- ✨ Clean, type-safe Python API
- 🌍 Support for 1600+ languages (English: `eng_Latn`, Filipino: `tgl_Latn`)
- 🚀 Multiple model sizes (300M to 7B parameters)
- 📦 Easy installation with pip
- 🛠️ Command-line interface included
- ⏱️ **Automatic chunking** for audio files longer than 30 seconds
- 🔊 **Automatic resampling** to 16kHz
- 💪 Batch processing support

## Installation

### Standard Installation

```bash
pip install -e .
```

### Development Installation

```bash
pip install -e ".[dev]"
```

### AMD/ROCm (RX 6600) Installation

To use this on your AMD card, you **must** install the ROCm version of PyTorch:

```bash
# Uninstall existing torch
pip uninstall torch torchio

# Install pytorch-rocm (use the version matching your ROCm install)
# See: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
pip install torch torchaudio --index-url [https://download.pytorch.org/whl/rocm5.6](https://download.pytorch.org/whl/rocm5.6)
```

---

## Quick Start

### Python API

**Note for 8GB VRAM:** You **must** use a smaller model. `1B` is recommended. `7B` will cause an Out-of-Memory error.

```python
from omniasr_headless import OmniASRAdapter
import sys

# Initialize adapter. 
# For 8GB VRAM, "1B" is required. 
# Use "cuda" to target your AMD GPU via ROCm.
try:
    adapter = OmniASRAdapter(model_size="1B", device="cuda")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Ensure pytorch-rocm is installed and 8GB VRAM is free.")
    sys.exit(1)

# --- Transcribe a LONG audio file ---
# This file is 2 minutes long, but the adapter will
# automatically chunk it into 30s pieces and join the results.
print("Transcribing long file (long_audio.wav)...")
text = adapter.transcribe("long_audio.wav", language="eng_Latn", batch_size=2)
print("\n--- Full Transcript (long_audio.wav) ---")
print(text)


# --- Transcribe Filipino/Tagalog audio ---
print("\nTranscribing Tagalog file (tagalog.wav)...")
text_tgl = adapter.transcribe("tagalog.wav", language="tgl_Latn", batch_size=2)
print("\n--- Full Transcript (tagalog.wav) ---")
print(text_tgl)
```

### Command Line (AMD/ROCm)

You **must** prefix your commands with `HSA_OVERRIDE_GFX_VERSION=10.3.0` to enable your RX 6600.

```bash
# Transcribe English audio (using 1B model for 8GB VRAM)
HSA_OVERRIDE_GFX_VERSION=10.3.0 omniasr long_audio.wav --model 1B --device cuda --language eng_Latn

# Transcribe Filipino audio
HSA_OVERRIDE_GFX_VERSION=10.3.0 omniasr tagalog.wav --model 1B --device cuda --language tgl_Latn

# Process multiple files (batch size 2 is good for 8GB VRAM)
HSA_OVERRIDE_GFX_VERSION=10.3.0 omniasr file1.wav file2.wav --model 1B --device cuda --batch-size 2

# Output as JSON
HSA_OVERRIDE_GFX_VERSION=10.3.0 omniasr audio.wav --model 1B --device cuda --json > output.json
```

---

## Model Sizes

| Model | Parameters | Speed | Accuracy | VRAM (Approx) |
|-------|-----------|-------|----------|---------------|
| 300M | 317M | ⚡⚡⚡ Fast | Good | ~2-3 GB |
| 1B | 965M | ⚡⚡ Faster | Better | ~4-6 GB |
| 3B | 3.1B | ⚡ Moderate | Great | ~10-12 GB |
| 7B | 6.5B | 🐌 Slower | **Best** | ~18-20 GB |

**Recommendation:**
* **For 8GB VRAM (RX 6600):** You **must** use the `1B` model. You may be able to use `300M` for faster speeds.
* **For Max Accuracy:** `7B` is best, but requires a high-end GPU (18GB+ VRAM).

---

## Requirements

- Python 3.8+
- PyTorch (standard or ROCm build)
- `libsndfile` for audio support:
  - **Mac**: `brew install libsndfile`
  - **Ubuntu**: `sudo apt-get install libsndfile1`
  - **Fedora**: `sudo dnf install libsndfile`
  - **Windows**: May need additional setup

### AMD GPU (ROCm) Setup
- **ROCm Libraries:** Must be installed (e.g., `sudo dnf install rocm-hip-devel`)
- **PyTorch Build:** Must be `pytorch-rocm`
- **VRAM:** ~4-6GB VRAM for the `1B` model.
- **Run Variable:** You must set `HSA_OVERRIDE_GFX_VERSION=10.3.0` every time you run the script.

---

## Limitations

- 🗣️ **No Automatic Mixed-Language:** The model requires a *single* language code (`eng_Latn`, `tgl_Latn`, etc.) for each audio file or segment. It cannot detect and transcribe multiple languages mixed together in one chunk. See "Mixed Language Audio" section.
- 🔪 **Chunking Artifacts:** Automatic chunking of long files may occasionally cut a word in half at the 30-second mark. This is a trade-off for handling arbitrarily long files.
- 💾 **Large Model Downloads:** The "1B" model requires ~4GB of disk space. The "7B" model requires ~29GB.
