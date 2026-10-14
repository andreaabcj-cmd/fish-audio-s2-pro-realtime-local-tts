# Raw data

All measurements were taken on the machine described in the [guide](../README.md#1-test-machine). Utterance texts and voice names are omitted
(voices are anonymised as `voice_A` / `voice_B`).

| file | what it is |
|---|---|
| `tts_streaming_per_utterance.csv` | 114 rows: 3 GGUFs (Q8_0/Q6_K/Q4_K) × 2 voices × 19 utterances (incl. emotion-tag variants). First audio, generation time, audio length, ratio. audio.cpp streaming (SSE). |
| `tts_offline_per_utterance.csv` | same, non-streaming (the only mode release v0.9.1 supports for Fish). "first audio" = total time. |
| `start_delay_needed.json` | for 6 answers per GGUF (Q8_0, Q4_K): the playback start delay (s) needed to never run dry — one request vs. first sentence + rest. |
| `pipeline_cushion_runs_raw.json` | real LLM + real TTS + simulated sound card, cushion/queue settings `0.65/60` and `0.8/60`. The `0.5/6` and `1.1/60` lines in the guide came from console output of the same script and were not saved here. |
| `llm_prompt_reread_per_turn.csv` | per-turn prompt re-read time of the chat model over replayed 8-turn conversations. `run_label`: `base` = before; `ckpt256*` = checkpoint-spacing env var (no effect; the `b/c` runs were with `OLLAMA_DEBUG=1`, 7× slower); `warm1` = extra stable-prefix request; `real1`, `final_sem_perfil`, `padrao_ollama` = the same trick through our real prompt builder (`padrao_ollama` = default Ollama settings). |
