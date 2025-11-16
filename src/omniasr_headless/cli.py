"""
Command-line interface for Omnilingual ASR headless adapter
"""

import argparse
import sys
from pathlib import Path
import json

from omniasr_headless.adapter import OmniASRAdapter


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Omnilingual ASR - Transcribe audio in 1600+ languages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transcribe English audio
  omniasr audio.wav --language eng_Latn
  
  # Transcribe Filipino/Tagalog audio
  omniasr audio.wav --language tgl_Latn
  
  # Transcribe multiple files
  omniasr file1.wav file2.wav file3.wav --language eng_Latn
  
  # Use smaller/faster model
  omniasr audio.wav --model 1B --language eng_Latn
  
  # Output as JSON
  omniasr audio.wav --language eng_Latn --json
  
  # List supported languages
  omniasr --list-languages
        """
    )
    
    parser.add_argument(
        "audio_files",
        nargs="*",
        help="Audio file(s) to transcribe"
    )
    
    parser.add_argument(
        "-l", "--language",
        default="eng_Latn",
        help="Language code (e.g., eng_Latn, tgl_Latn). Default: eng_Latn"
    )
    
    parser.add_argument(
        "-m", "--model",
        choices=["300M", "1B", "3B", "7B"],
        default="7B",
        help="Model size. 7B is most accurate. Default: 7B"
    )
    
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=2,
        help="Batch size for processing. Default: 2"
    )
    
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device to use. Default: auto"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Output file (default: print to stdout)"
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )
    
    parser.add_argument(
        "--list-languages",
        action="store_true",
        help="List common language codes and exit"
    )
    
    parser.add_argument(
        "--model-info",
        action="store_true",
        help="Show model information and exit"
    )
    
    args = parser.parse_args()
    
    # Handle info commands
    if args.list_languages:
        adapter = OmniASRAdapter(model_size="7B")
        langs = adapter.get_supported_languages()
        print("\nCommonly Used Language Codes:")
        print("-" * 50)
        for name, code in langs.items():
            print(f"{name:<25} {code}")
        print("\nNote: 1600+ languages supported. See docs for full list.")
        return 0
    
    if args.model_info:
        adapter = OmniASRAdapter(model_size=args.model)
        info = adapter.model_info
        print("\nModel Information:")
        print("-" * 50)
        for key, value in info.items():
            print(f"{key}: {value}")
        return 0
    
    # Validate audio files
    if not args.audio_files:
        parser.print_help()
        print("\nError: No audio files specified", file=sys.stderr)
        return 1
    
    # Check files exist
    for file_path in args.audio_files:
        if not Path(file_path).exists():
            print(f"Error: File not found: {file_path}", file=sys.stderr)
            return 1
    
    try:
        # Initialize adapter
        print(f"Loading model ({args.model})...", file=sys.stderr)
        device = None if args.device == "auto" else args.device
        adapter = OmniASRAdapter(model_size=args.model, device=device)
        
        # Transcribe
        print(f"Transcribing {len(args.audio_files)} file(s)...", file=sys.stderr)
        transcriptions = adapter.transcribe(
            args.audio_files,
            language=args.language,
            batch_size=args.batch_size
        )
        
        # Format output
        if isinstance(transcriptions, str):
            transcriptions = [transcriptions]
        
        if args.json:
            output = json.dumps({
                "files": args.audio_files,
                "language": args.language,
                "transcriptions": transcriptions
            }, indent=2, ensure_ascii=False)
        else:
            output = ""
            for i, (file_path, text) in enumerate(zip(args.audio_files, transcriptions), 1):
                if len(args.audio_files) > 1:
                    output += f"\n[{i}] {file_path}\n"
                output += text + "\n"
        
        # Write output
        if args.output:
            Path(args.output).write_text(output, encoding="utf-8")
            print(f"Output written to: {args.output}", file=sys.stderr)
        else:
            print(output)
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
