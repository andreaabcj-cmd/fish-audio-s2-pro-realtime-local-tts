# Research notes: running Fish Audio S2 Pro in real time on a 16 GB GPU

What we read before measuring anything (13 Oct 2026), condensed. We stand on a lot of other people's work — thank you.
Labels: **[confirmed]** = official source, maintainer, or a published measurement · **[community]** = a user's report ·
**[estimate]** = our own back-of-envelope reasoning (measured later where noted in the [guide](README.md)).

## Why it is hard
Each audio frame (1/21.5 s) needs a read of the big "slow" model (≈4B params, ≈8 GB in BF16) **and** nine passes of the small "fast" model (≈400M).
A 16 GB consumer card reads ~448 GB/s, so in BF16 the theoretical ceiling is around 30 frames/s while real time needs 21.5. **[estimate]**
Quantizing roughly doubles that ceiling — which is why GGUF Q8/Q6/Q4 mattered so much on this card.

## Which runtime streams for real
* **audio.cpp** (C++/GGML, Windows + CUDA) **[confirmed by project docs]**: `--mode streaming` with an *incremental* codec (keeps codec state, sends only new audio);
  HTTP `/v1/audio/speech` with `"stream_format": "sse"`; options like `fish_audio.codec_weight_type`, `--max-tokens`, `--text-chunk-size`.
  <https://github.com/0xShug0/audio.cpp/blob/main/docs/models/fish_audio.md> — an RTX 5090 user reports 3.1–3.4× faster than real time:
  <https://github.com/fishaudio/fish-speech/issues/1308>.
  **Our finding: the v0.9.1 release binary rejects streaming for Fish; build from source ≥ `b0feeff` ([guide §2](README.md)).**
* **SGLang-Omni** **[confirmed]**: real streaming in windows (`stream_stride=40`, `stream_followup_stride=45`, `stream_overlap_tokens=20`,
  `stream_crossfade_samples=512` in `stages.py`); `torch.compile` for the small model is only enabled on sm_90 (H100/H200), so on RTX 50xx (sm_120) it runs uncompiled.
  <https://github.com/sgl-project/sglang-omni/blob/main/sglang_omni/models/fishaudio_s2_pro/stages.py> ·
  PR #391 (codec ≈0.6 GB + audio decoder ≈1.4 GB are outside SGLang's memory accounting): <https://github.com/sgl-project/sglang-omni/pull/391>.
  A community RTX 5090 recipe reaches 0.36 s first audio, RTF 0.48, but needs ~24.6 GB: <https://huggingface.co/neko-legends/fish-audio-s2-realtime-optimized-5090>.
  In our tests on 16 GB (BF16, mem 0.55): first audio 1.1–1.3 s, RTF ≈1.15–1.24, ~8 GB; raising memory to 0.70 made it *worse* (RTF 1.8 — the card spilled).
* **s2.cpp** **[community]**: GGUF down to Q2, but "exact" streaming re-decodes the beginning (higher latency). <https://github.com/rodrigomatta/s2.cpp>
* **vLLM-Omni** supports it but peaks ~49 GB on an A800. Not an option here.
* The reference server of `fish-speech` was not usable for real time in our tests (BF16 + compile: 15.8 tok/s, RTF 1.88, ~15 GB; INT8 + compile: 25.3 tok/s, RTF 0.87, 5.5 GB model, 11 GB peak decoding 30 s at once).

## Quantization
* **GGUF in audio.cpp** **[community, with table]**: Q8_0 6.32 GB, Q6_K 5.44 GB, Q4_K 4.50 GB, keeping the small "fast" decoder and the codec at full precision.
  On an 8 GB laptop 4070 (~256 GB/s) RTF ≈ 1.25 / 1.15 / 1.15 with clean audio. **Lesson: quantizing the small model ruins the sound.**
  Recipe: <https://github.com/0xShug0/audio.cpp/blob/main/docs/proposals/fish_audio_quantization_recipe.md> ·
  files: <https://huggingface.co/audio-cpp/audio.cpp-gguf> · <https://huggingface.co/rodrigomt/s2-pro-gguf>.
* **SGLang-Omni does not accept a quantized S2 Pro** **[confirmed]**: PR #1511 says S2 Pro "never forwards quant_config" and now refuses `--quantization fp8`. <https://github.com/sgl-project/sglang-omni/pull/1511>
* **FP8 weight-only checkpoints** (several authors): ~4.7–6.2 GB of weights but the math is done in BF16 (unpacked) — saves memory, not speed. <https://huggingface.co/AEmotionStudio/fish-speech-s2-pro-fp8>
* **NF4 / bitsandbytes INT8 / w4a16** **[community]**: slowness and >21 GB use reported (the 32 768-position cache is to blame).
  <https://github.com/fishaudio/fish-speech/issues/1168> · <https://huggingface.co/scarxity/fish-speech-s2-pro-nf4>
* **MXFP8 from Fish itself** **[confirmed]**: production uses MXFP8 on RTX 5090 via `fish-scales-ops` (Apache-2.0), kernels only, Linux + an exact PyTorch build; a layout mistake once gave 53% logit error that speed numbers did not reveal.
  <https://fish.audio/blog/tts-inference-optimization-s2-pro-free/> · <https://github.com/fishaudio/fish-scales-ops>

## Codec
The codec is causal: it can decode in chunks while keeping state (audio.cpp) or in windows with 20 tokens of overlap + crossfade (SGLang-Omni). Decoding 30 s at once is what gave a ~11 GB peak in our first PyTorch runs.
Limiting `max_seq_len` to 4096 saves ~4.5 GB of cache in the official repo (<https://github.com/fishaudio/fish-speech/pull/1306>). A sibling model (Drama 3) decodes in fp16 by default, a hint that fp16 may be safe (<https://github.com/sgl-project/sglang-omni/blob/main/docs/cookbook/fishaudio_drama3.md>); audio.cpp keeps the codec in F16.

## Smaller models
There is no "S2 mini". S1-mini (0.5B, CC-BY-NC-SA) is the previous generation: a bit worse (WER 0.011 vs 0.008) and uses `(angry)` style tags. <https://huggingface.co/fishaudio/s1-mini>

## Emotion tags
S2/S2 Pro use square brackets with free text, e.g. `[whisper]`, `[laughing]`, `[excited]`, `[sad]`, `[surprised]`, `[pause]`, `[low voice]`, `[very excited]` — 15 000+ tags. <https://huggingface.co/fishaudio/s2-pro> · <https://docs.fish.audio/api-reference/emotion-reference>
Official tips: put the tag at the start of the sentence, combine (`[sad][whispering]`), add intensity (`[slightly sad]`), at most ~3 emotions per sentence. Spanish is "tier 2" (good, below English/Chinese/Japanese).
**In our listening tests the emotion already came from the text itself; we did not need tags** (and did not evaluate them). A calm reference recording pulls everything toward calm.

## RTX 50xx (sm_120) notes
* SGLang-Omni: "no kernel image" crash (flash-attention kernel only for Hopper), open issue: <https://github.com/sgl-project/sglang-omni/issues/160>; workaround from the 5090 recipe `FISH_S2PRO_DISABLE_FLASH_ATTN_KVCACHE=1` + `sdpa`.
* Official repo needs PyTorch with CUDA 12.8+; with `--compile` on RTX 50 one report says `torch._inductor.config.triton.cudagraph_trees = False` fixes it: <https://github.com/fishaudio/fish-speech/issues/919>.
* audio.cpp on Windows: CUDA ≥ 12.4 (for sm_120 in practice 12.8+). We used 13.3.1 and `-CudaArchitectures "120a-real"`.

## What we did with it
We tried audio.cpp first (Q8 → Q6 → Q4). It won on memory and simplicity (single native binary, no Python/WSL for the model), and Q4 won on smoothness. The SGLang-Omni and INT8 paths stayed as plan B and were never needed.
