from __future__ import annotations

import math
import statistics
import time
import wave
from dataclasses import dataclass
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

MAX_AUDIO_BYTES = 10 * 1024 * 1024
SUPPORTED_CONTENT_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/vnd.wave",
}

app = FastAPI(
    title="DuoAccent Accent Scoring API",
    version="1.0.0",
    description="Scores short WAV audio clips for accent clarity across pronunciation, stress, liaisons, and intonation.",
)


@dataclass(frozen=True)
class AudioFeatures:
    duration_seconds: float
    sample_rate: int
    rms: float
    silence_ratio: float
    peak_ratio: float
    zero_crossing_rate: float
    energy_variation: float
    pitch_variation: float
    voiced_ratio: float


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/score-accent")
async def score_accent(file: UploadFile = File(...)) -> JSONResponse:
    started = time.perf_counter()

    if file.content_type not in SUPPORTED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Unsupported audio format. Please upload a WAV file.",
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")
    if len(content) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio file must be 10 MB or smaller.")

    try:
        features = extract_wav_features(content)
    except wave.Error as exc:
        raise HTTPException(status_code=400, detail="Invalid WAV audio file.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    scores = score_features(features)
    overall = round(sum(scores.values()) / len(scores))
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

    return JSONResponse(
        {
            "score": overall,
            "feedback": {
                "pronunciation": build_feedback(
                    scores["pronunciation"],
                    "Sound clarity is strong with stable articulation.",
                    "Some sounds are unclear; record in a quieter place and exaggerate consonant endings.",
                ),
                "stress": build_feedback(
                    scores["stress"],
                    "Stress patterns show useful emphasis between syllables and phrases.",
                    "Stress sounds flat; vary loudness on important words and reduce filler pauses.",
                ),
                "liaisons": build_feedback(
                    scores["liaisons"],
                    "Word linking is natural with limited disruptive silence.",
                    "Connected speech is choppy; practice linking final consonants into following vowel sounds.",
                ),
                "intonation": build_feedback(
                    scores["intonation"],
                    "Pitch movement gives the sentence a natural rise and fall.",
                    "Intonation is limited; add pitch movement for questions, lists, and emphasis.",
                ),
            },
            "parameters": scores,
            "metadata": {
                "duration_seconds": round(features.duration_seconds, 2),
                "sample_rate": features.sample_rate,
                "processing_time_ms": elapsed_ms,
                "latency_budget_ms": 10000,
            },
        }
    )


def extract_wav_features(content: bytes) -> AudioFeatures:
    with wave.open(BytesIO(content)) as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frame_count = wav.getnframes()
        frames = wav.readframes(frame_count)

    if channels < 1:
        raise ValueError("Audio must have at least one channel.")
    if sample_width not in {1, 2, 4}:
        raise ValueError("Only 8-bit, 16-bit, or 32-bit PCM WAV files are supported.")
    if sample_rate <= 0 or frame_count <= 0:
        raise ValueError("Audio must contain valid samples.")

    duration = frame_count / sample_rate
    if duration < 1:
        raise ValueError("Audio must be at least 1 second long.")
    if duration > 30:
        raise ValueError("Audio must be 30 seconds or shorter.")

    mono_samples = decode_mono_samples(frames, sample_width, channels)
    sample_count = len(mono_samples)
    if sample_count == 0 or max(abs(sample) for sample in mono_samples) == 0:
        raise ValueError("Audio must not be silent.")

    window_size = max(int(sample_rate * 0.05), 1)
    windows = [
        mono_samples[start : start + window_size]
        for start in range(0, len(mono_samples) - window_size + 1, window_size)
    ]
    if not windows:
        windows = [mono_samples]

    max_amplitude = float((1 << (8 * sample_width - 1)) - 1)
    energies = [rms(window) / max_amplitude for window in windows]
    silence_threshold = max(percentile(energies, 20) * 0.8, 0.01)
    voiced_indexes = [index for index, energy in enumerate(energies) if energy > silence_threshold]

    rms_value = rms(mono_samples) / max_amplitude
    silence_ratio = 1 - (len(voiced_indexes) / len(windows))
    peak_ratio = clipped_sample_ratio(mono_samples, max_amplitude)
    zero_crossing_rate = zero_crossings(mono_samples) / max(sample_count - 1, 1)
    voiced_energies = [energies[index] for index in voiced_indexes]
    energy_variation = coefficient_of_variation(voiced_energies)
    pitch_variation = estimate_pitch_variation(
        [windows[index] for index in voiced_indexes[:200]], sample_rate
    )
    voiced_ratio = len(voiced_indexes) / len(windows)

    return AudioFeatures(
        duration_seconds=duration,
        sample_rate=sample_rate,
        rms=rms_value,
        silence_ratio=silence_ratio,
        peak_ratio=peak_ratio,
        zero_crossing_rate=zero_crossing_rate,
        energy_variation=energy_variation,
        pitch_variation=pitch_variation,
        voiced_ratio=voiced_ratio,
    )


def decode_mono_samples(frames: bytes, sample_width: int, channels: int) -> list[float]:
    raw_samples = list(iter_samples(frames, sample_width))
    if channels == 1:
        return raw_samples
    return [
        statistics.fmean(raw_samples[index : index + channels])
        for index in range(0, len(raw_samples) - channels + 1, channels)
    ]


def iter_samples(frames: bytes, sample_width: int):
    signed = sample_width != 1
    for start in range(0, len(frames), sample_width):
        chunk = frames[start : start + sample_width]
        if len(chunk) == sample_width:
            value = int.from_bytes(chunk, "little", signed=signed)
            yield value - 128 if sample_width == 1 else value


def rms(samples: list[float]) -> float:
    if not samples:
        return 0.0
    return math.sqrt(statistics.fmean(sample * sample for sample in samples))


def clipped_sample_ratio(samples: list[float], max_amplitude: float) -> float:
    if not samples:
        return 0.0
    threshold = max_amplitude * 0.95
    clipped = sum(1 for sample in samples if abs(sample) >= threshold)
    return clipped / len(samples)


def zero_crossings(samples: list[float]) -> int:
    crossings = 0
    previous_negative: bool | None = None
    for sample in samples:
        negative = sample < 0
        if previous_negative is not None and negative != previous_negative:
            crossings += 1
        previous_negative = negative
    return crossings


def coefficient_of_variation(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = statistics.fmean(values)
    if mean == 0:
        return 0.0
    return statistics.pstdev(values) / mean


def estimate_pitch_variation(windows: list[list[float]], sample_rate: int) -> float:
    pitches: list[float] = []
    for window in windows:
        crossings = crossing_indexes(window)
        if len(crossings) > 1:
            gaps = [b - a for a, b in zip(crossings, crossings[1:])]
            mean_gap = statistics.fmean(gaps)
            if mean_gap > 0:
                frequency = sample_rate / (2 * mean_gap)
                if 60 <= frequency <= 400:
                    pitches.append(frequency)
    return coefficient_of_variation(pitches)


def crossing_indexes(samples: list[float]) -> list[int]:
    indexes: list[int] = []
    previous_negative: bool | None = None
    for index, sample in enumerate(samples):
        negative = sample < 0
        if previous_negative is not None and negative != previous_negative:
            indexes.append(index)
        previous_negative = negative
    return indexes


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percent / 100
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[int(index)]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def score_features(features: AudioFeatures) -> dict[str, int]:
    pronunciation = weighted_score(
        closeness(features.zero_crossing_rate, 0.08, 0.08),
        closeness(features.rms, 0.12, 0.12),
        1 - clamp(features.peak_ratio * 8),
    )
    stress = weighted_score(
        closeness(features.energy_variation, 0.55, 0.55),
        1 - clamp(features.silence_ratio * 1.2),
        closeness(features.voiced_ratio, 0.72, 0.5),
    )
    liaisons = weighted_score(
        1 - clamp(features.silence_ratio * 1.5),
        closeness(features.voiced_ratio, 0.78, 0.55),
        closeness(features.energy_variation, 0.45, 0.7),
    )
    intonation = weighted_score(
        closeness(features.pitch_variation, 0.18, 0.22),
        closeness(features.energy_variation, 0.5, 0.65),
        1 - clamp(features.silence_ratio),
    )

    return {
        "pronunciation": pronunciation,
        "stress": stress,
        "liaisons": liaisons,
        "intonation": intonation,
    }


def weighted_score(*values: float) -> int:
    return round(clamp(sum(values) / len(values)) * 100)


def closeness(value: float, target: float, tolerance: float) -> float:
    if tolerance <= 0:
        return 0.0
    return 1 - clamp(abs(value - target) / tolerance)


def clamp(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return min(max(value, 0.0), 1.0)


def build_feedback(score: int, positive: str, improvement: str) -> str:
    if score >= 75:
        return positive
    if score >= 50:
        return f"{positive} {improvement}"
    return improvement
