# DuoAccent Accent Scoring API

A FastAPI implementation of Task B from the take-home assignment. It exposes `POST /score-accent`, accepts a WAV audio clip, and returns a 0-100 accent score with feedback for pronunciation, stress, liaisons, and intonation.

## Requirements covered

- Working REST API
- `POST /score-accent` endpoint
- Audio upload validation
- Correct HTTP errors for invalid input
- JSON response with overall score and feedback across four parameters
- Server-side processing designed to finish under 10 seconds
- Dockerfile and Docker Compose setup

## Run with Docker Compose

```bash
docker compose up --build
```

The API will be available at:

```text
http://localhost:8000
```

Interactive API docs:

```text
http://localhost:8000/docs
```

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

## Tests

```bash
python -m pytest -q
```

The test suite covers health checks, valid WAV scoring, unsupported content type, empty upload, invalid WAV, silent audio, too-short audio, and too-long audio.

## Example request

```bash
curl -X POST "http://localhost:8000/score-accent" \
  -F "file=@sample.wav;type=audio/wav"
```

Example response:

```json
{
  "score": 73,
  "feedback": {
    "pronunciation": "Sound clarity is strong with stable articulation.",
    "stress": "Stress patterns show useful emphasis between syllables and phrases. Stress sounds flat; vary loudness on important words and reduce filler pauses.",
    "liaisons": "Word linking is natural with limited disruptive silence.",
    "intonation": "Intonation is limited; add pitch movement for questions, lists, and emphasis."
  },
  "parameters": {
    "pronunciation": 82,
    "stress": 68,
    "liaisons": 76,
    "intonation": 65
  },
  "metadata": {
    "duration_seconds": 4.2,
    "sample_rate": 16000,
    "processing_time_ms": 12.45,
    "latency_budget_ms": 10000
  }
}
```

## Validation and errors

- `415 Unsupported Media Type`: file is not a WAV upload
- `400 Bad Request`: empty file, invalid WAV, silent audio, unsupported PCM width, or audio shorter than 1 second
- `413 Payload Too Large`: file is larger than 10 MB
- `400 Bad Request`: audio is longer than 30 seconds

Only PCM WAV files are supported in this implementation.

## Design decisions

This implementation uses a deterministic local scoring model rather than a remote speech or LLM API. The goal is to provide a reliable API contract, fast response times, and explainable scoring behavior within the take-home scope.

The scoring extracts lightweight audio features:

- RMS energy and clipping ratio for pronunciation clarity
- Energy variation and voiced/silent windows for stress
- Silence ratio and voiced continuity for liaisons
- Estimated pitch variation for intonation

The endpoint returns both an overall score and individual parameter scores so clients can show detailed learner feedback.

## Latency approach

The hard requirement is under 10 seconds from request receipt to full JSON response. This API avoids network calls and processes only short WAV clips up to 30 seconds and 10 MB. Feature extraction uses lightweight standard-library audio operations and caps pitch estimation to the first 200 voiced windows, keeping runtime predictable.

The response includes `metadata.processing_time_ms` so latency can be observed per request.

With more time, I would add benchmark tests that fail if p95 latency exceeds a chosen threshold, and I would run them in CI against representative audio samples.

## What I would improve with more time

- Support MP3/M4A by adding a safe transcoding step with ffmpeg
- Use a real ASR/phoneme alignment model for pronunciation scoring
- Compare learner speech against a target sentence or reference recording
- Add automated tests with generated WAV fixtures
- Add structured logs and request IDs
- Calibrate scores against human-rated accent samples
