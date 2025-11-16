from typing import List, Dict, Union, Optional, Literal, cast, Sequence
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as T
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline


ModelSize = Literal["300M", "1B", "3B", "7B"]
TARGET_SR = 16000
CHUNK_DURATION_SEC = 30
CHUNK_SAMPLES = CHUNK_DURATION_SEC * TARGET_SR


class OmniASRAdapter:
    
    MODEL_CARDS = {
        "300M": "omniASR_W2V_300M",
        "1B": "omniASR_W2V_1B",
        "3B": "omniASR_W2V_3B",
        "7B": "omniASR_LLM_7B",
    }
    
    def __init__(
        self, 
        model_size: ModelSize = "7B",
        device: Optional[str] = None,
    ):
        self.model_size = model_size
        self.model_card = self.MODEL_CARDS[model_size]
        
        self.pipeline = ASRInferencePipeline(
            model_card=self.model_card,
            device=device
        )
    
    def _load_and_chunk_audio(
        self, 
        audio_input: Union[str, Path, np.ndarray], 
        input_sr: Optional[int] = None
    ) -> List[np.ndarray]:
        
        if isinstance(audio_input, (str, Path)):
            audio, sr = sf.read(audio_input)
        elif isinstance(audio_input, np.ndarray):
            if input_sr is None:
                raise ValueError("sample_rate is required when audio is a numpy array")
            audio, sr = audio_input, input_sr
        
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
            
        audio_resampled = audio
        if sr != TARGET_SR:
            try:
                audio_tensor = torch.from_numpy(audio).float()
                resampler = T.Resample(orig_freq=sr, new_freq=TARGET_SR)
                audio_resampled = resampler(audio_tensor).numpy()
            except Exception as e:
                raise RuntimeError(f"Error resampling audio. Ensure 'libsndfile' is installed. Error: {e}")

        duration_samples = audio_resampled.shape[0]
        
        if duration_samples <= CHUNK_SAMPLES:
            return [audio_resampled]
        else:
            chunks = []
            for i in range(0, duration_samples, CHUNK_SAMPLES):
                chunk = audio_resampled[i : i + CHUNK_SAMPLES]
                chunks.append(chunk)
            return chunks

    def transcribe(
        self,
        audio: Union[str, Path, np.ndarray, Sequence[Union[str, Path, np.ndarray]]],
        language: Union[str, List[str]] = "eng_Latn",
        batch_size: int = 2,
        sample_rate: Optional[int] = None
    ) -> Union[str, List[str]]:
        
        is_single: bool
        audio_list: List[Union[str, Path, np.ndarray]]

        if isinstance(audio, (str, Path, np.ndarray)):
            is_single = True
            audio_list = [audio]
        else:
            is_single = False
            audio_list = list(audio)
            
        if isinstance(language, str):
            lang_list = [language] * len(audio_list)
        else:
            lang_list = language
            
        if len(lang_list) != len(audio_list):
            raise ValueError("Number of language codes must match number of audio files")

        all_audio_dicts = []
        input_to_chunk_map = []
        processed_lang_list = []

        for i, aud_input in enumerate(audio_list):
            audio_chunks_np = self._load_and_chunk_audio(aud_input, sample_rate)
            
            for chunk_np in audio_chunks_np:
                all_audio_dicts.append({
                    "waveform": chunk_np,
                    "sample_rate": TARGET_SR
                })
            
            input_to_chunk_map.append(len(audio_chunks_np))
            processed_lang_list.extend([lang_list[i]] * len(audio_chunks_np))
        
        all_transcriptions = self.pipeline.transcribe(
            all_audio_dicts,
            lang=processed_lang_list,
            batch_size=batch_size
        )
        
        final_results = []
        current_index = 0
        for num_chunks in input_to_chunk_map:
            chunks_for_this_file = all_transcriptions[current_index : current_index + num_chunks]
            joined_text = " ".join(chunks_for_this_file)
            final_results.append(joined_text)
            current_index += num_chunks

        return final_results[0] if is_single else final_results
    
    def transcribe_file(
        self,
        file_path: Union[str, Path],
        language: str = "eng_Latn"
    ) -> str:
        return cast(str, self.transcribe(file_path, language=language))
    
    def transcribe_mixed_languages(
        self,
        audio_files: List[Union[str, Path]],
        languages: List[str],
        batch_size: int = 2
    ) -> List[str]:
        return cast(List[str], self.transcribe(audio_files, language=languages, batch_size=batch_size))
    
    def get_supported_languages(self) -> Dict[str, str]:
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
        }
    
    @property
    def model_info(self) -> Dict[str, Union[str, int]]:
        return {
            "model_size": self.model_size,
            "model_card": self.model_card,
            "supported_languages": "1600+",
            "max_audio_length": "40 seconds",
        }
