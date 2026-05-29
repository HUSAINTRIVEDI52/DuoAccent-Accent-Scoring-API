import io
import math
import struct
import wave

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def make_wav(duration_seconds=2.0, sample_rate=16000, amplitude=12000, frequency=180):
    buffer = io.BytesIO()
    frame_count = int(sample_rate * duration_seconds)
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(
            b"".join(
                struct.pack(
                    "<h",
                    int(amplitude * math.sin(2 * math.pi * frequency * index / sample_rate)),
                )
                for index in range(frame_count)
            )
        )
    buffer.seek(0)
    return buffer.getvalue()


def post_audio(content, filename="sample.wav", content_type="audio/wav"):
    return client.post(
        "/score-accent",
        files={"file": (filename, io.BytesIO(content), content_type)},
    )


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_valid_wav_score():
    response = post_audio(make_wav())
    body = response.json()

    assert response.status_code == 200
    assert 0 <= body["score"] <= 100
    assert set(body["parameters"]) == {"pronunciation", "stress", "liaisons", "intonation"}
    assert set(body["feedback"]) == {"pronunciation", "stress", "liaisons", "intonation"}
    assert body["metadata"]["processing_time_ms"] < body["metadata"]["latency_budget_ms"]


def test_unsupported_content_type():
    response = post_audio(b"not audio", filename="sample.mp3", content_type="audio/mpeg")
    assert response.status_code == 415


def test_empty_upload():
    response = post_audio(b"")
    assert response.status_code == 400


def test_invalid_wav():
    response = post_audio(b"not a wav")
    assert response.status_code == 400


def test_silent_audio():
    response = post_audio(make_wav(amplitude=0))
    assert response.status_code == 400
    assert response.json()["detail"] == "Audio must not be silent."


def test_too_short_audio():
    response = post_audio(make_wav(duration_seconds=0.5))
    assert response.status_code == 400
    assert response.json()["detail"] == "Audio must be at least 1 second long."


def test_too_long_audio():
    response = post_audio(make_wav(duration_seconds=31.0))
    assert response.status_code == 400
    assert response.json()["detail"] == "Audio must be 30 seconds or shorter."
