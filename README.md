# RealTimeTranslateTTS

> **English** | [中文](README-cn.md)

An Android real-time speech translation app (simultaneous interpretation) supporting microphone and system media audio input with speech recognition, translation, and text-to-speech output.

---

## Latest Updates (2026-04)

- **Claude API** — Translation engine + AI refinement, supporting the full Anthropic model lineup (Opus 4 / Sonnet 4 / Haiku 4.5)
- **Volcano Engine TTS** — ByteDance high-quality neural voices (12+ Chinese, 4 English, 2 Japanese)
- **GPU / NPU acceleration** — Unified ONNX provider toggle (CPU / NNAPI / XNNPACK) across all Sherpa modules and offline translation, with automatic CPU fallback on init failure
- **API key management** — Full CRUD for 6 credential types (OpenAI / Groq / DeepL / Claude / Volcano App ID + Token); server configs now editable in-place
- **Connectivity tests** — Added Claude translation test + Volcano TTS test; one-tap "Test All"
- **Expanded model lists** — OpenAI (GPT-4.1 / o4-mini / o3-mini), Groq (DeepSeek R1 70B), Claude full lineup, local (DeepSeek R1 7B)
- **v1.2.3** — Sherpa streaming ASR, unified offline TTS pipeline, Kokoro v1.1 Chinese voices
- **SWR dual-channel translation** — Fast path displayed immediately; quality upgrade replaces it asynchronously
- **Context-enhanced translation** — Latency mode, background context, domain hints, glossary injection

---

## Features

### Speech Recognition (ASR)

| Engine | Description |
|--------|-------------|
| System ASR | Android native recognition, zero setup |
| Vosk (offline) | Local English recognition, no network needed |
| Sherpa Streaming ASR | OnlineRecognizer with real-time partial + endpoint results; multiple models (Zipformer / Paraformer / CTC / NeMo) |
| OpenAI Whisper API | Cloud Whisper via OpenAI, high accuracy |
| Groq Whisper API | Cloud Whisper via Groq, low latency |
| GPT-4o Transcribe | OpenAI's latest transcription model |
| Local Whisper (Sherpa-ONNX) | On-device Whisper with Silero VAD, fully offline |

### Translation Engines

| Engine | Description |
|--------|-------------|
| Google MLKit (offline) | On-device translation, no network |
| OpenAI GPT | OpenAI Chat API (GPT-4o / 4.1 / 4.1-mini / 4.1-nano) |
| Groq LLM | Low-latency LLM (Llama 3.3 70B / DeepSeek R1, free tier) |
| DeepL | High-quality translation API |
| **Claude API** | Anthropic Messages API (Sonnet 4 / Haiku 4.5 / Opus 4 / 3.5 Sonnet / 3.5 Haiku) |
| Local Server | Self-hosted LLM (Ollama / LM Studio — custom URL + model name) |
| Opus-MT / NLLB (offline) | On-device ONNX translation models, fully private |

### Text-to-Speech (TTS)

| Engine | Description |
|--------|-------------|
| Microsoft Edge TTS | Neural voices (free, high quality) |
| Android System TTS | Native system voice |
| Google Translate TTS | Google Translate audio endpoint |
| OpenAI TTS | OpenAI speech API (alloy / nova / shimmer, etc.) |
| Sherpa-ONNX Offline TTS | Unified pipeline: VITS / Matcha / Kokoro / Kitten |
| **Volcano Engine TTS** | ByteDance neural voices (Chinese: Cancan, Qingcang, etc.) |

### Translation Enhancements

- **SWR (Stale-While-Revalidate)** — Fast translation displayed immediately; higher-quality result replaces it in the background
- **Latency modes** — Realtime / Balanced / Quality
- **Background context** — Optional context paragraph to guide the model
- **Domain routing** — auto / general / meeting / medical / customer_support / game
- **Glossary injection** — Per-domain term injection for consistent terminology

### AI Refinement

Post-translation polish using a fast LLM. Six provider options: Groq / OpenAI / **Claude** / On-device (Termux+Ollama) / LAN server / Off. Full model selection for each.

### Glossary Management

- **Built-in glossaries** — General, meeting, medical, customer support, gaming
- **Downloadable glossaries** — Open-source term packs with license validation
- **User import** — CSV/TSV upload (source, target columns)
- **Priority merge** — User > Downloaded > Built-in

### Media Audio Translation

Capture system/app audio via MediaProjection API (Android 10+) for real-time transcription and translation — no microphone needed.

### Floating Overlay

When the app goes to background, a floating translation window continues recording and translating. Results sync back to main-screen history.

### GPU / NPU Acceleration

- Three ONNX execution providers: **CPU** (default) / **NNAPI** (Android GPU/NPU/DSP) / **XNNPACK** (optimized CPU)
- Covers all Sherpa ASR, TTS, VAD, speech enhancement, language detection, and on-device translation (ORT)
- Automatic fallback: if the preferred provider fails to initialize, the engine retries with CPU
- Actual provider logged at init time for easy on-device verification

### API Key Management & Connectivity Tests

- **Key manager** — CRUD for 6 credential types with masked display and one-tap delete
- **Server configs** — Editable URL + model name for translation and refinement servers
- **Connectivity tests** — Step-by-step checks (DNS → Connection → Auth → Functional) for all configured APIs; supports both "Test All" and individual service tests

### Other

- Translation history (session grouping, search, rename)
- TTS echo suppression in microphone mode
- Smart ASR filtering (filler words, noise, music, echo)
- Latency dashboard (ASR / Translation / Refinement / TTS per-stage timing)
- Device metrics (CPU / memory / battery / temperature)
- Audio device selection (input / output)

---

## Requirements

- **Android 8.0+** (API 26+); media audio capture requires Android 10+
- Microphone permission (`RECORD_AUDIO`)
- Overlay permission (`SYSTEM_ALERT_WINDOW`, optional — for floating window)
- Media projection permission (optional — for system audio capture)
- Network permission (when using online APIs)

---

## Architecture

```
MainActivity                 ← Main UI (Jetpack Compose)
├── ASR Engines
│   ├── System / Vosk / SherpaStreamingAsr / SherpaWhisperAsr
│   └── WhisperApiAsr (OpenAI / Groq / GPT-4o)
├── TranslationPipeline      ← Concurrent translation + SWR upgrade + ordered delivery
│   ├── TranslationContext   (latencyMode / background / domainHint)
│   ├── GlossaryManager      (domain routing & term injection)
│   ├── TranslationEngine    (MLKit / LLM / DeepL / Claude / Server / ONNX)
│   ├── QualityEngine        (optional background quality channel)
│   └── TranslationRefiner   (Groq / OpenAI / Claude / Local LLM)
├── TTS Consumer
│   └── Edge / System / Google / OpenAI / SherpaOnnx / Volcano
├── AccelerationConfig       ← ONNX provider config & fallback chain
├── MediaCaptureService      ← Foreground service: system audio capture + ASR
└── FloatingTranslateService ← Foreground service: background overlay translation
```

---

## Key Dependencies

| Library | Purpose |
|---------|---------|
| [Vosk](https://alphacephei.com/vosk/) | Offline speech recognition |
| [Sherpa-ONNX](https://github.com/k2-fsa/sherpa-onnx) | On-device Whisper ASR + TTS |
| [Google MLKit](https://developers.google.com/ml-kit/language/translation) | Offline translation |
| [ONNX Runtime](https://onnxruntime.ai/) | On-device inference (NNAPI / XNNPACK) |
| [OkHttp](https://square.github.io/okhttp/) | HTTP & WebSocket |
| Jetpack Compose | UI framework |
| Kotlin Coroutines | Async concurrency |
