# Fish Audio S2 Pro as a *real-time* local voice on a 16 GB GPU — next to a 12B LLM (Windows, RTX 5060 Ti)

<p align="center"><img src="assets/ali_tchau.gif" alt="Ali, the mascot of the assistant, waving goodbye (seamless loop)" width="190"></p>

Field notes from building a fully local voice assistant (Spanish, voice cloning from a short reference + transcript).
Everything below was measured on **one machine**, in October 2026. Where something was *not* measured, it says so.

> 🔊 **Listen to the samples** on the [project page](https://andreaabcj-cmd.github.io/fish-s2-local-voice-notes/#en-listen) or in [`samples/`](samples/) (Spanish; Q4_K vs Q8_0; emotion from text vs inline tags).

> **TL;DR**
> * Fish S2 Pro runs fine in real time on a 16 GB consumer GPU **alongside a 12B chat model**, with the *Q4_K* GGUF.
> * First audio arrives in **~0.15–0.35 s** with streaming (0.5 s through our HTTP adapter), but generation runs at only **~1.0× the audio duration**
>   (Q4 median; Q8 ~1.12×) — so you need a small playback buffer (we use 0.8 s) or you get clicks.
> * **audio.cpp release 0.9.1 cannot stream Fish** even though its README says so. Build from source (≥ commit `b0feeff`, 13 Oct 2026).
> * Separate win for the LLM side: warming the *stable prefix* of the prompt cut the per-answer prompt re-read from **3.4 s → ~1.2 s**.

## 1. Test machine

| | |
|---|---|
| CPU / RAM | Ryzen 7 5800X3D / 64 GB |
| GPU | GeForce RTX 5060 Ti **16 GB** (Blackwell, compute 12.0), Windows 11 Pro, normal desktop running (~2 GB of VRAM used by apps) |
| Chat model | `gemma4:12b` in Ollama 0.35.1 (8 GB file incl. vision projector + MTP drafter), KV cache `q8_0`, `num_ctx 8192`, batch 512 |
| TTS | Fish Audio S2 Pro GGUF via [audio.cpp](https://github.com/0xShug0/audio.cpp) (CUDA), Q4_K / Q6_K / Q8_0 |
| Speech-to-text, etc. | other models resident too; not covered here |

## 2. Getting audio.cpp to stream Fish (the part that cost us the most time)

**Symptom:** with the Windows CUDA release `v0.9.1` (7 Oct 2026), a server config with `"mode": "streaming"` for the Fish model fails at start:

```
audiocpp_server failed: Fish Audio S2-Pro supports offline TTS sessions
```

The `fish_audio.md` doc and README table say streaming is supported, but that landed in the source tree **after** 0.9.1.
(`--mode streaming` in `audiocpp_cli` fails the same way.) Offline mode works but the first sound only arrives when the *whole* utterance is
done — ~30 s of waiting for a 30 s answer (RTF 0.9–1.1) — useless for conversation.

**Fix: build the server from source** (commit `b0feeff`, 13 Oct 2026 or newer). Nothing needs to be installed on `C:`:

1. **MSVC**: Visual Studio 2022 *Build Tools* with the C++ workload (we had 14.44). CMake comes with it.
2. **CUDA without the 3 GB installer**: NVIDIA publishes the toolkit as redistributable zips:
   `https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.3.1.json` lists them. We downloaded the Windows-x86_64 zips for
   `cuda_nvcc, cuda_cudart, cccl, cuda_crt, cuda_nvrtc, libnvvm, libcublas, libcufft, cuda_profiler_api, libnvjitlink, libnvfatbin,
   libnvptxcompiler, cuda_nvtx, cuda_cuxxfilt, libcurand, libcusparse, libcusolver` and merged them into one folder (3.9 GB total — a
   superset; we did not try to find the minimum). Note `cccl` and `libnvvm` have different manifest keys than you might expect.
3. **Build** (≈ 20–25 min, ~1200 steps). The repo's script needs a PowerShell execution-policy bypass:

   ```powershell
   $env:CUDA_PATH = "D:\cuda\toolkit"; $env:CUDAToolkit_ROOT = $env:CUDA_PATH
   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_windows.ps1 `
       -Preset windows-cuda-release -Target audiocpp_server -CudaArchitectures "120a-real"
   ```
   `120a-real` is for Blackwell cards (RTX 50xx) only — it makes the build faster but the binary will not run on other GPUs.
4. The result is a **single `audiocpp_server.exe`**, no DLLs. Copy the DLLs (`ggml*.dll`, `cublas*.dll`, `cudart*.dll`, …) from the
   release zip's `bin` folder next to it. `--version` should print your build, and the Fish streaming config should now start.

**GGUF files:** Q8_0 comes from the audio.cpp GGUF repo (`audio-cpp/audio.cpp-gguf`); Q6_K and Q4_K are produced with the project's own
recipe — read `docs/proposals/fish_audio_quantization_recipe.md` in the source tree. The key point is that the small "fast acoustic
decoder" tensors must stay at original precision (`--keep-type 'model_weights/fast_*=orig'`) and the token metadata sidecar must be
extracted from the Q8 file; naive quantization gives garbled audio. File sizes: Q8 6.3 GB, Q6 5.4 GB, Q4 4.5 GB.

## 3. Server config and request

`server.json` (voice cloning = a preset with a reference wav **and its exact transcript**):

```json
{ "host": "127.0.0.1", "port": 18190, "backend": "cuda", "device": 0, "busy_timeout_ms": 600000,
  "models": [{ "id": "fish", "family": "fish_audio", "task": "tts", "mode": "streaming",
               "path": "D:/models/Fish-Audio-S2-Pro-GGUF/fish-audio-s2-pro-q4_k.gguf",
               "model_spec_override": "D:/audiocpp/src/model_specs",
               "session_options": { "fish_audio.reference_cache_slots": "4" },
               "voice_presets": { "my_voice": { "voice_ref": "D:/voices/ref.wav", "reference_text": "exact transcript of ref.wav" } } }] }
```
(`model_spec_override` is what we ran with; we did not test whether a new enough binary needs it.)

Request: `POST /v1/audio/speech` with `{"model":"fish","input":"...","voice":"my_voice","response_format":"pcm","stream_format":"sse"}`.
You get SSE events; `speech.audio.delta` carries **base64 PCM16 mono at 44.1 kHz**. Closing the HTTP connection stops generation
(measured: the next request started speaking in 0.65 s after an interruption).

Our adapter ([`fish_tts_adapter.py`](fish_tts_adapter.py), ~200 lines, standard library + numpy/scipy/requests) turns that into
"POST text → raw 24 kHz PCM16 as it is generated", and also: starts/stops audio.cpp, `/sleep` + `/wake` to free VRAM, and a Windows *Job Object* so
audio.cpp dies with the adapter (an orphaned server keeps ~5 GB of VRAM). Tested: first byte 0.5 s.
Two details worth copying: streaming 44.1→24 kHz with `resample_poly` **in blocks with context on both sides** (bit-identical to a one-shot
resample, verified with several chunk sizes — no seams), and a soft limiter instead of hard clipping.

## 4. Results

### 4.1 TTS alone (GPU otherwise idle), streaming
38 utterances per GGUF (2 voices × 19 texts, 2–34 s of speech). "Generation ÷ audio" > 1 means slower than real time.

| GGUF | first audio | generation time ÷ audio time (range, median) | peak VRAM incl. ~2 GB desktop |
|---|---|---|---|
| Q8_0 | 0.15–0.31 s | 0.98–1.27, median 1.12 | 8.4 GB |
| Q6_K | 0.17–0.35 s | 0.98–1.19, median 1.09 | 7.7 GB |
| **Q4_K** | 0.14–0.31 s | 0.85–1.08, **median 1.00** | **7.0 GB** |

How much *start delay* you need so playback never runs dry, for ~19 s answers (measured from the SSE arrival times):
Q8 1.0–2.5 s, Q4 0.2–0.5 s. So **Q4 is not only the smallest, it is the one that plays smoothly** — on our ears (Spanish, one listener) all
three sounded equally good; the listener chose Q4 for efficiency.
Emotional tone came out of the text itself (punctuation, words); we did **not** need inline emotion tags and did not evaluate them.

### 4.2 Full pipeline (real LLM + real TTS, simulated sound card that counts under-runs)
Q4, short conversational answers (~8–10 s of speech each). "Cushion" = audio buffered before playback starts; "queue" = how far generation may run ahead.

| cushion / queue | audible gaps | speech tested |
|---|---|---|
| 0.5 s / 6 | **1** | 48 s |
| 0.65 s / 60 | 0 | 52 s |
| **0.8 s / 60** (our setting) | 0 / 0 / 0 | 50 + 51 + 48 s |
| 1.1 s / 60 | 0 | 52 s |

With Q6 and the small cushion, a real session logged **8 gaps in one 11 s answer** — those are the "little clicks" a listener reports.
Time from the request to the first sound: **~1.9–2.4 s median** across those runs (≈0.7 s of that is the LLM writing its first sentence).

### 4.3 VRAM
`gemma4:12b` (≈8.5 GB dedicated) + Fish Q4 + Windows desktop: **~14.1 GB of 16 GB used** (it was ~15.7 GB with the TTS we replaced).

## 5. Bonus: making the LLM start answering sooner (Ollama / llama.cpp, sliding-window models)

Before: every answer re-read the whole ~5 000-token prompt (**3.4 s**) although only the last turn had changed. With `gemma4` (sliding-window
attention) llama.cpp can only reuse a cached prefix through *context checkpoints*, and it only creates them at the **end of a request** — i.e.
after the part that changes every turn, so they were never usable. (The server log shows `forcing full prompt re-processing due to lack of cache data`.)

What worked (Ollama 0.35.1, no environment variables needed): after each answer, once speech has finished, send **one extra request containing only the
stable prefix** (system prompt + conversation history, *without* the volatile "current context" block and *without* the new user message) with
`num_predict: 1`. The checkpoint now lands exactly where the next turn's prompt diverges.
Keep tools, `num_ctx`, `num_batch`, `keep_alive` identical to the real requests or the model reloads. Put everything that changes per turn (clock, retrieved
memories) **after** the history.

* Measured on a replayed 8-turn conversation: re-read before answering **3.4 s → ~1.2 s** (warm-up itself ~0.3–0.7 s while idle); in live use 1.3–1.7 s.
* Things that did **not** help: `LLAMA_ARG_CHECKPOINT_MIN_SPACING_NT=256` (no effect); `LLAMA_ARG_SWA_FULL=1` works but costs ~2.2 GB of VRAM.
* Moving the user profile *before* the history would save another ~0.6 s, but with this 12B model it made recall of the *current conversation* worse
  (asked "what did I say her name was?" it answered from the profile: 0/6 correct vs 6/6). Check for this kind of regression if you reorder prompts.
* `OLLAMA_DEBUG=1` made prompt reads ~7× slower in our runs (23 s vs 3 s); don't leave it on while measuring.
* Not tested: newer Ollama (0.40.x) — its notes were mostly about Apple/MLX.

## 6. Gotchas, quickly
* Don't trust "streaming supported" in a README: grep the release for it, or just try a config with `"mode":"streaming"`.
* Streaming gives *time to first audio*, not speed. If generation is ≥ 1.0× audio duration you still need a buffer.
* Kill orphaned `audiocpp_server.exe` processes — each holds ~5 GB of VRAM.
* First start after boot was slow (≈57 s, cold disk read); afterwards 10–20 s. Waking an adapter that put audio.cpp to sleep took ≈10 s.
* In PowerShell scripts blocked by policy, use `-ExecutionPolicy Bypass` for that command only.
* If you run Git Bash, backslashes inside heredocs get eaten (we corrupted a Windows path in a doc that way).

## 7. Not measured / limits
Other languages; other voices; word-error-rate or any objective quality metric (only one human listener); behaviour while gaming or rendering at the same time;
long sessions (hours); audio.cpp versions newer than `b0feeff`; non-Blackwell GPUs.

## 8. License note
The Fish Audio S2 Pro model card restricts the weights to **non-commercial** use — check the current terms before building anything beyond personal use.
audio.cpp, llama.cpp and Ollama have their own licenses.

## Credits
This was a **team effort**:

* **André** ([@andreaabcj-cmd](https://github.com/andreaabcj-cmd)) — product direction, the hardware, every decision, and the listening tests (the only human ears behind the "sounds good" claims).
* **Claude** (Anthropic, via Claude Code) — engineering, measurements, research, and this write-up. All numbers come from scripts run on the machine above; raw data is in [`data/`](data/).
* **The open-source community**, whose work made it possible and whose findings we read before measuring anything ([RESEARCH.md](RESEARCH.md)): the
  [audio.cpp](https://github.com/0xShug0/audio.cpp) project and its Fish quantization recipe, [Fish Audio](https://fish.audio) and the `fish-speech` issue threads,
  [s2.cpp](https://github.com/rodrigomatta/s2.cpp) and the GGUF publishers on Hugging Face, SGLang-Omni, llama.cpp (and the maintainers who documented the sliding-window checkpoint behaviour), Ollama, and everyone who posted benchmarks and bug reports.

Raw measurements: [`data/`](data/) · What we read first: [RESEARCH.md](RESEARCH.md) · Example adapter: [`fish_tts_adapter.py`](fish_tts_adapter.py).
