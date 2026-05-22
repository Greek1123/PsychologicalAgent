from __future__ import annotations

import io
import math
import struct
import wave
from dataclasses import dataclass, field


@dataclass(slots=True)
class AudioSignal:
    modality: str
    filename: str
    content_type: str | None
    byte_size: int
    analysis_available: bool
    format: str | None = None
    duration_seconds: float | None = None
    sample_rate_hz: int | None = None
    channels: int | None = None
    sample_width_bits: int | None = None
    rms_energy: float | None = None
    peak_amplitude: float | None = None
    silence_ratio: float | None = None
    analysis_notes: list[str] = field(default_factory=list)


def analyze_audio_signal(*, file_bytes: bytes, filename: str, content_type: str | None) -> AudioSignal:
    signal = AudioSignal(
        modality="audio",
        filename=filename,
        content_type=content_type,
        byte_size=len(file_bytes),
        analysis_available=False,
    )
    if not file_bytes:
        signal.analysis_notes.append("empty_audio")
        return signal

    if not _looks_like_wav(filename, content_type):
        signal.analysis_notes.append("basic_features_unavailable_for_non_wav")
        return signal

    try:
        with wave.open(io.BytesIO(file_bytes), "rb") as wav:
            channels = wav.getnchannels()
            sample_rate = wav.getframerate()
            sample_width = wav.getsampwidth()
            frame_count = wav.getnframes()
            frames = wav.readframes(frame_count)
    except (wave.Error, EOFError):
        signal.analysis_notes.append("wav_parse_failed")
        return signal

    signal.format = "wav"
    signal.sample_rate_hz = sample_rate
    signal.channels = channels
    signal.sample_width_bits = sample_width * 8
    signal.duration_seconds = round(frame_count / sample_rate, 3) if sample_rate else None

    samples = _decode_pcm_samples(frames, sample_width)
    if not samples:
        signal.analysis_notes.append("pcm_samples_unavailable")
        return signal

    max_abs_value = float((2 ** (sample_width * 8 - 1)) - 1) if sample_width > 1 else 128.0
    abs_samples = [abs(sample) for sample in samples]
    peak = max(abs_samples) / max_abs_value if max_abs_value else 0.0
    rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples)) / max_abs_value
    silence_threshold = max_abs_value * 0.02
    silence_ratio = sum(1 for value in abs_samples if value <= silence_threshold) / len(abs_samples)

    signal.analysis_available = True
    signal.rms_energy = round(min(rms, 1.0), 4)
    signal.peak_amplitude = round(min(peak, 1.0), 4)
    signal.silence_ratio = round(silence_ratio, 4)
    signal.analysis_notes.append("wav_basic_signal_features_extracted")
    return signal


def _looks_like_wav(filename: str, content_type: str | None) -> bool:
    lower_name = filename.lower()
    lower_type = (content_type or "").lower()
    return lower_name.endswith(".wav") or lower_type in {"audio/wav", "audio/x-wav", "audio/wave"}


def _decode_pcm_samples(frames: bytes, sample_width: int) -> list[int]:
    if not frames:
        return []
    if sample_width == 1:
        return [byte - 128 for byte in frames]
    if sample_width == 2:
        count = len(frames) // 2
        return list(struct.unpack(f"<{count}h", frames[: count * 2]))
    if sample_width == 4:
        count = len(frames) // 4
        return list(struct.unpack(f"<{count}i", frames[: count * 4]))
    return []
