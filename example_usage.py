import sys
from pathlib import Path
from omniasr_headless import OmniASRAdapter

# --- Configuration ---
#
# IMPORTANT: For 8GB RX 6600 GPU, you MUST use a smaller model.
# "1B" is recommended. "7B" will cause an Out-of-Memory error.
#
# Use "cuda" as the device. The pytorch-rocm build
# maps "cuda" to AMD GPU.
#
MODEL_TO_USE = "1B" 
DEVICE = "cuda"
BATCH_SIZE = 2  # Keep this low (1 or 2) for 8GB VRAM

def main():
    print(f"Initializing adapter (Model: {MODEL_TO_USE}, Device: {DEVICE})...")
    
    try:
        # Initialize the adapter
        adapter = OmniASRAdapter(model_size=MODEL_TO_USE, device=DEVICE)
        print("Adapter loaded successfully.")
    
    except Exception as e:
        print(f"\n--- FATAL ERROR ---")
        print(f"Could not load model: {e}")
        print("\n--- Troubleshooting (AMD/ROCm) ---")
        print("1. Did you install 'pytorch-rocm'?")
        print("2. Did you run this script with 'HSA_OVERRIDE_GFX_VERSION=10.3.0'?")
        print(f"   Example: HSA_OVERRIDE_GFX_VERSION=10.3.0 python {sys.argv[0]}")
        print("3. Do you have enough VRAM? The '1B' model needs ~4-6GB.")
        sys.exit(1)

    # Create dummy audio files for testing
    # We will create one short file and one long (60s) file
    try:
        import numpy as np
        import soundfile as sf
        
        SR = 16000
        
        # Create a 2-second dummy file
        short_audio = np.random.randn(SR * 2)
        sf.write("short_test.wav", short_audio, SR)

        # Create a 60-second dummy file (2x the 30s limit)
        long_audio = np.random.randn(SR * 60)
        sf.write("long_test.wav", long_audio, SR)
        
        print("Created 'short_test.wav' (2s) and 'long_test.wav' (60s).")

    except ImportError:
        print("Install numpy and soundfile to create dummy files: pip install numpy soundfile")
        print("Skipping dummy file creation.")
    except Exception as e:
        print(f"Error creating dummy files: {e}")


    # --- 1. Transcribe a single LONG file ---
    # The adapter will automatically chunk 'long_test.wav'
    print("\n--- Test 1: Transcribing single long file (60s) ---")
    if Path("long_test.wav").exists():
        try:
            transcript = adapter.transcribe(
                "long_test.wav",
                language="eng_Latn",
                batch_size=BATCH_SIZE
            )
            print(f"Transcript (long_test.wav):\n{transcript}")
        except Exception as e:
            print(f"Error during transcription: {e}")
    else:
        print("Skipping Test 1 (long_test.wav not found).")


    # --- 2. Transcribe multiple files in a batch ---
    # This will process both files, batching all chunks together
    print("\n--- Test 2: Transcribing multiple files in batch ---")
    files_to_process = ["short_test.wav", "long_test.wav"]
    
    if all(Path(f).exists() for f in files_to_process):
        try:
            transcripts = adapter.transcribe(
                files_to_process,
                language="eng_Latn",
                batch_size=BATCH_SIZE
            )
            print("Batch transcription complete.")
            for f, t in zip(files_to_process, transcripts):
                print(f"\nFile: {f}\nTranscript: {t}\n")
        except Exception as e:
            print(f"Error during batch transcription: {e}")
    else:
        print("Skipping Test 2 (test files not found).")

if __name__ == "__main__":
    main()
