"""
Omnilingual ASR Headless Adapter

A clean, type-safe Python interface for Meta's Omnilingual ASR supporting 1600+ languages
"""

from omniasr_headless.adapter import OmniASRAdapter, load_audio

__version__ = "0.1.0"
__all__ = ["OmniASRAdapter", "load_audio"]
