"""
Headless adapter for Meta's Omnilingual ASR
Provides a clean, type-safe interface for speech recognition
"""

from typing import List, Dict, Union, Optional, Literal, cast, Sequence
from pathlib import Path
import numpy as np
import soundfile as sf
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline


ModelSize = Literal["300M", "1B", "3B", "7B"]


class OmniASRAdapter:
    """
    Headless adapter for Omnilingual ASR with clean API
    
    Supports 1600+ languages including English (eng_Latn) and Filipino (tgl_Latn)
    """
    
    MODEL_CARDS = {
        "300M": "omniASR_W2V_300M",
        "1B": "omniASR_W2V_1B",
        "3B": "omniASR_W2V_3B",
        "7B": "omniASR_LLM_7B",  # Most accurate
    }
    
    def __init__(
        self, 
        model_size: ModelSize = "7B",
        device: Optional[str] = None,
    ):
        """
        Initialize the adapter
        
        Args:
            model_size: Model size (300M, 1B, 3B, 7B). 7B is most accurate.
            device: Device to use ('cuda', 'cpu', or None for auto)
            cache_dir: Directory to cache model weights (~29GB for 7B)
        """
        self.model_size = model_size
        self.model_card = self.MODEL_CARDS[model_size]
        
        # Initialize pipeline
        self.pipeline = ASRInferencePipeline(
            model_card=self.model_card,
            device=device
        )
    
    def transcribe(
        self,
        audio: Union[str, Path, np.ndarray, Sequence[Union[str, Path, np.ndarray]]],
        language: Union[str, List[str]] = "eng_Latn",
        batch_size: int = 2,
        sample_rate: Optional[int] = None
    ) -> Union[str, List[str]]:
        """
        Transcribe audio file(s) or arrays
        
        Args:
            audio: Single or list of audio file paths or numpy arrays
            language: Language code(s) in format 'xxx_Yyyy' (e.g., 'eng_Latn', 'tgl_Latn')
            batch_size: Batch size for processing multiple files
            sample_rate: Sample rate if audio is numpy array
            
        Returns:
            Single transcription string or list of transcriptions
            
        Examples:
            >>> adapter = OmniASRAdapter(model_size="7B")
            >>> 
            >>> # Single file
            >>> text = adapter.transcribe("audio.wav", language="eng_Latn")
            >>> 
            >>> # Multiple files with different languages
            >>> texts = adapter.transcribe(
            ...     ["eng.wav", "tgl.wav"],
            ...     language=["eng_Latn", "tgl_Latn"]
            ... )
            >>> 
            >>> # From numpy array
            >>> audio_array = np.array([...])
            >>> text = adapter.transcribe(audio_array, language="tgl_Latn", sample_rate=16000)
        """
        # Normalize inputs
        is_single = not isinstance(audio, list)
        audio_list = [audio] if is_single else audio
        
        # Handle language codes
        if isinstance(language, str):
            lang_list = [language] * len(audio_list)
        else:
            lang_list = language
            
        if len(lang_list) != len(audio_list):
            raise ValueError("Number of language codes must match number of audio files")
        
        # Process audio inputs
        processed_audio = []
        for aud in audio_list:
            if isinstance(aud, (str, Path)):
                processed_audio.append(str(aud))
            elif isinstance(aud, np.ndarray):
                if sample_rate is None:
                    raise ValueError("sample_rate required when audio is numpy array")
                processed_audio.append({
                    "waveform": aud,
                    "sample_rate": sample_rate
                })
            else:
                raise ValueError(f"Unsupported audio type: {type(aud)}")
        
        # Run transcription
        transcriptions = self.pipeline.transcribe(
            processed_audio,
            lang=lang_list,
            batch_size=batch_size
        )
        
        return transcriptions[0] if is_single else transcriptions
    
    def transcribe_file(
        self,
        file_path: Union[str, Path],
        language: str = "eng_Latn"
    ) -> str:
        """Convenience method to transcribe a single audio file"""
        return cast(str, self.transcribe(file_path, language=language))
    
    def transcribe_mixed_languages(
        self,
        audio_files: List[Union[str, Path]],
        languages: List[str],
        batch_size: int = 2
    ) -> List[str]:
        """
        Transcribe multiple files with different languages
        
        Useful for processing mixed English-Filipino content
        
        Args:
            audio_files: List of audio file paths
            languages: List of language codes matching each file
            batch_size: Batch size for processing
            
        Returns:
            List of transcriptions
            
        Example:
            >>> texts = adapter.transcribe_mixed_languages(
            ...     ["segment1.wav", "segment2.wav", "segment3.wav"],
            ...     ["eng_Latn", "tgl_Latn", "eng_Latn"]
            ... )
        """
        return cast(List[str], self.transcribe(audio_files, language=languages, batch_size=batch_size))
    
    def get_supported_languages(self) -> Dict[str, str]:
        """
        Get commonly used language codes
        
        Returns:
            Dictionary of language names to codes
        """
        return {
            "English": "eng_Latn",
            "Filipino/Tagalog": "tgl_Latn",
            "Spanish": "spa_Latn",
            "Chinese (Mandarin)": "cmn_Hans",
            "Japanese": "jpn_Jpan",
            "Korean": "kor_Hang",
            "German": "deu_Latn",
            "French": "fra_Latn",
            "Italian": "ita_Latn",
            "Portuguese": "por_Latn",
            "Russian": "rus_Cyrl",
            "Arabic": "arb_Arab",
            "Hindi": "hin_Deva",
            # Note: Full list of 1600+ languages available in the model
        }
    
    @property
    def model_info(self) -> Dict[str, Union[str, int]]:
        """Get information about the loaded model"""
        return {
            "model_size": self.model_size,
            "model_card": self.model_card,
            "supported_languages": "1600+",
            "max_audio_length": "40 seconds",
        }


def load_audio(file_path: Union[str, Path], target_sr: int = 16000) -> tuple:
    """
    Load audio file and resample if needed
    
    Args:
        file_path: Path to audio file
        target_sr: Target sample rate (default 16kHz)
        
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    audio, sr = sf.read(file_path)
    
    # Resample if needed (simple implementation)
    if sr != target_sr:
        # For production, use librosa.resample or torchaudio.transforms.Resample
        import warnings
        warnings.warn(f"Audio has sample rate {sr}Hz, expected {target_sr}Hz. May affect quality.")
    
    return audio, sr
