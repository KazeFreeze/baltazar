"""
Example usage of Omnilingual ASR Headless Adapter
"""

from omniasr_headless import OmniASRAdapter

def main():
    # Initialize adapter with 7B model (most accurate)
    print("Loading model (this may take a while on first run)...")
    adapter = OmniASRAdapter(model_size="7B")
    
    # Example 1: Transcribe English audio
    print("\n=== Example 1: English Audio ===")
    text = adapter.transcribe("english_audio.wav", language="eng_Latn")
    print(f"Transcription: {text}")
    
    # Example 2: Transcribe Filipino/Tagalog audio
    print("\n=== Example 2: Filipino/Tagalog Audio ===")
    text = adapter.transcribe("tagalog_audio.wav", language="tgl_Latn")
    print(f"Transcription: {text}")
    
    # Example 3: Process multiple files
    print("\n=== Example 3: Batch Processing ===")
    files = ["audio1.wav", "audio2.wav", "audio3.wav"]
    texts = adapter.transcribe(files, language="eng_Latn", batch_size=2)
    for i, (file, text) in enumerate(zip(files, texts), 1):
        print(f"[{i}] {file}: {text}")
    
    # Example 4: Mixed languages (English + Filipino)
    print("\n=== Example 4: Mixed English + Filipino ===")
    # Assume you've segmented the audio into separate files
    segments = ["segment1_eng.wav", "segment2_tgl.wav", "segment3_eng.wav"]
    languages = ["eng_Latn", "tgl_Latn", "eng_Latn"]
    
    texts = adapter.transcribe_mixed_languages(segments, languages)
    for segment, lang, text in zip(segments, languages, texts):
        print(f"{segment} ({lang}): {text}")
    
    # Example 5: Show supported languages
    print("\n=== Example 5: Common Language Codes ===")
    langs = adapter.get_supported_languages()
    for name, code in list(langs.items())[:5]:
        print(f"{name}: {code}")
    
    # Example 6: Model information
    print("\n=== Example 6: Model Info ===")
    info = adapter.model_info
    for key, value in info.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
