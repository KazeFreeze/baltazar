# Omnilingual ASR Headless Adapter

A clean, industry-standard Python adapter for Meta's Omnilingual ASR supporting **1600+ languages** including English and Filipino/Tagalog.

## Features

- ✨ Clean, type-safe Python API
- 🌍 Support for 1600+ languages (English: `eng_Latn`, Filipino: `tgl_Latn`)
- 🚀 Multiple model sizes (300M to 7B parameters)
- 📦 Easy installation with pip
- 🛠️ Command-line interface included
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

### What is `pip install -e .`?

The `-e` flag installs the package in **editable mode** (also called "development mode"):
- Changes to source code are immediately reflected without reinstalling
- The package is symlinked to your local directory instead of copied to site-packages
- Perfect for development, testing, and local modifications
- You can edit the code and test changes instantly

## Quick Start

### Python API

```python
from omniasr_headless import OmniASRAdapter

# Initialize adapter (downloads ~29GB model on first run)
adapter = OmniASRAdapter(model_size="7B")

# Transcribe English audio
text = adapter.transcribe("audio.wav", language="eng_Latn")
print(text)

# Transcribe Filipino/Tagalog audio
text = adapter.transcribe("tagalog.wav", language="tgl_Latn")
print(text)

# Process multiple files with different languages
texts = adapter.transcribe_mixed_languages(
    audio_files=["english.wav", "tagalog.wav", "english2.wav"],
    languages=["eng_Latn", "tgl_Latn", "eng_Latn"]
)

# Use smaller/faster model
adapter_fast = OmniASRAdapter(model_size="1B")
text = adapter_fast.transcribe("audio.wav", language="eng_Latn")
```

### Command Line

```bash
# Transcribe English audio
omniasr audio.wav --language eng_Latn

# Transcribe Filipino audio
omniasr tagalog.wav --language tgl_Latn

# Process multiple files
omniasr file1.wav file2.wav file3.wav --language eng_Latn

# Use faster model (1B parameters)
omniasr audio.wav --model 1B --language eng_Latn

# Output as JSON
omniasr audio.wav --language eng_Latn --json > output.json

# List common language codes
omniasr --list-languages

# Show model information
omniasr --model-info
```

## Model Sizes

| Model | Parameters | Speed | Accuracy | Use Case |
|-------|-----------|-------|----------|----------|
| 300M | 317M | ⚡⚡⚡ Fast | Good | Low-power devices |
| 1B | 965M | ⚡⚡ Faster | Better | Quick processing |
| 3B | 3.1B | ⚡ Moderate | Great | Balanced |
| 7B | 6.5B | 🐌 Slower | **Best** | Maximum accuracy |

**Recommendation**: Use 7B for best results (<10% error rate on 78% of languages)

## Language Codes

Common language codes (format: `xxx_Yyyy`):

| Language | Code |
|----------|------|
| English | `eng_Latn` |
| Filipino/Tagalog | `tgl_Latn` |
| Spanish | `spa_Latn` |
| Chinese (Mandarin) | `cmn_Hans` |
| Japanese | `jpn_Jpan` |
| Korean | `kor_Hang` |
| German | `deu_Latn` |
| French | `fra_Latn` |

See [full language list](https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/models/wav2vec2_llama/lang_ids.py) for all 1600+ supported languages.

## Mixed Language Audio (English + Filipino)

For audio containing multiple languages:

1. **Segment First**: Split audio into single-language segments
2. **Process Separately**: Transcribe each segment with its language code
3. **Combine Results**: Merge transcriptions in order

```python
# Example workflow for mixed English-Filipino content
adapter = OmniASRAdapter(model_size="7B")

# Assume you've segmented the audio
segments = ["segment1.wav", "segment2.wav", "segment3.wav"]
languages = ["eng_Latn", "tgl_Latn", "eng_Latn"]

results = adapter.transcribe_mixed_languages(segments, languages)

# Combine results
full_transcript = " ".join(results)
```

## Requirements

- Python 3.8+
- CUDA-capable GPU recommended (18GB VRAM for 7B model)
- ~29GB disk space for model weights (7B)
- `libsndfile` for audio support:
  - **Mac**: `brew install libsndfile`
  - **Ubuntu**: `sudo apt-get install libsndfile1`
  - **Windows**: May need additional setup

## Advanced Usage

### From Numpy Array

```python
import numpy as np
from omniasr_headless import OmniASRAdapter

adapter = OmniASRAdapter(model_size="7B")

# Your audio as numpy array (16kHz recommended)
audio_array = np.array([...])  # Shape: (samples,)

text = adapter.transcribe(
    audio_array,
    language="tgl_Latn",
    sample_rate=16000
)
```

### Batch Processing

```python
from pathlib import Path
from omniasr_headless import OmniASRAdapter

adapter = OmniASRAdapter(model_size="7B")

# Process all WAV files in a directory
audio_dir = Path("audio_files")
audio_files = list(audio_dir.glob("*.wav"))

# Process in batches
transcriptions = adapter.transcribe(
    audio_files,
    language="eng_Latn",
    batch_size=4  # Adjust based on VRAM
)

# Save results
for file, text in zip(audio_files, transcriptions):
    output_file = file.with_suffix(".txt")
    output_file.write_text(text, encoding="utf-8")
```

## Limitations

- ⏱️ Audio files must be **≤40 seconds** (current implementation)
- 🔊 Best results with clean audio
- 🗣️ Single language per transcription call (no automatic language mixing)
- 💾 Large model downloads (7B model = ~29GB)

## Troubleshooting

### Out of Memory Error
```python
# Use smaller model or reduce batch size
adapter = OmniASRAdapter(model_size="1B")
results = adapter.transcribe(files, batch_size=1)
```

### Slow Transcription
```python
# Use smaller model
adapter = OmniASRAdapter(model_size="1B")
```

### Audio Format Issues
```bash
# Convert audio to WAV format first
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

## Project Structure

```
omniasr-headless/
├── src/
│   └── omniasr_headless/
│       ├── __init__.py      # Package initialization
│       ├── adapter.py       # Main adapter class
│       └── cli.py           # Command-line interface
├── tests/
│   └── test_adapter.py      # Unit tests
├── setup.py                 # Package configuration
├── README.md                # This file
├── requirements.txt         # Dependencies
└── .gitignore              # Git ignore rules
```

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

Apache 2.0 License (following Omnilingual ASR's license)

## Credits

Built on top of [Meta's Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr)

## Citation

If you use this adapter in your research:

```bibtex
@misc{omnilingualasr2025,
  title={{Omnilingual ASR}: Open-Source Multilingual Speech Recognition for 1600+ Languages},
  author={{Omnilingual ASR Team}},
  year={2025},
  url={https://ai.meta.com/research/publications/omnilingual-asr-open-source-multilingual-speech-recognition-for-1600-languages/}
}
```

## Support

- 📖 [Omnilingual ASR Docs](https://github.com/facebookresearch/omnilingual-asr)
- 🐛 [Report Issues](https://github.com/yourusername/omniasr-headless/issues)
- 💬 [Discussions](https://github.com/yourusername/omniasr-headless/discussions)
