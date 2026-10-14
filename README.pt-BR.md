# Fish Audio S2 Pro como voz local em *tempo real* numa GPU de 16 GB — junto de um LLM de 12B (Windows, RTX 5060 Ti)

<p align="center"><img src="assets/ali_tchau.gif" alt="Ali, a mascote do assistente, dando tchauzinho (laço sem emenda)" width="190"></p>

Notas de campo de quem montou um assistente de voz 100% local (espanhol, clonagem de voz a partir de uma referência curta + transcrição).
Tudo foi medido em **uma única máquina**, em outubro de 2026. O que *não* foi medido está dito como tal.

> 🔊 **Escute a voz da Ali (Chica) em espanhol, português e inglês** na [página do projeto](https://andreaabcj-cmd.github.io/fish-audio-s2-pro-realtime-local-tts/#pt-escute) ou em [`samples/`](samples/).

> **Resumo**
>
> * O Fish S2 Pro roda bem em tempo real numa GPU de consumo de 16 GB **junto de um modelo de chat de 12B**, usando o GGUF **Q4_K**.
> * O primeiro áudio sai em **~0,15–0,35 s** com streaming (0,5 s pelo nosso adaptador HTTP), mas a geração é só **~1,0× a duração do áudio**
>   (mediana do Q4; Q8 ~1,12×) — é preciso um pequeno colchão de reprodução (usamos 0,8 s), senão aparecem estalos.
> * **O audio.cpp release 0.9.1 não faz streaming com o Fish**, embora o README diga que sim. Compile do código-fonte (≥ commit `b0feeff`, 13/10/2026).
> * Ganho separado no LLM: "aquecer" o *prefixo estável* do prompt reduziu a releitura por resposta de **3,4 s → ~1,2 s**.

## 1. Máquina de teste

| | |
|---|---|
| CPU / RAM | Ryzen 7 5800X3D / 64 GB |
| GPU | GeForce RTX 5060 Ti **16 GB** (Blackwell, compute 12.0), Windows 11 Pro, desktop normal aberto (~2 GB de VRAM usados por apps) |
| Modelo de chat | `gemma4:12b` no Ollama 0.35.1 (arquivo de 8 GB com projetor de visão + rascunhador MTP), KV cache `q8_0`, `num_ctx 8192`, lote 512 |
| TTS | Fish Audio S2 Pro GGUF via [audio.cpp](https://github.com/0xShug0/audio.cpp) (CUDA), Q4_K / Q6_K / Q8_0 |

## 2. Fazer o audio.cpp dar streaming no Fish (o que mais custou tempo)

**Sintoma:** com o release `v0.9.1` do Windows CUDA (7/10/2026), uma config com `"mode": "streaming"` para o Fish falha ao iniciar:

```
audiocpp_server failed: Fish Audio S2-Pro supports offline TTS sessions
```

A documentação `fish_audio.md` e a tabela do README dizem que há streaming, mas isso entrou no código **depois** da 0.9.1
(`audiocpp_cli --mode streaming` falha igual). O modo offline funciona, mas o primeiro som só sai quando a fala *inteira* termina —
~30 s de espera para uma resposta de 30 s — inútil para conversa.

**Solução: compilar o servidor** (commit `b0feeff`, 13/10/2026 ou mais novo). Nada precisa ser instalado no `C:`:

1. **MSVC**: *Build Tools* do Visual Studio 2022 com o pacote C++ (usamos 14.44). O CMake vem junto.
2. **CUDA sem o instalador de 3 GB**: a NVIDIA publica o toolkit em zips redistribuíveis; o manifesto fica em
   `https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.3.1.json`. Baixamos os zips Windows-x86_64 de
   `cuda_nvcc, cuda_cudart, cccl, cuda_crt, cuda_nvrtc, libnvvm, libcublas, libcufft, cuda_profiler_api, libnvjitlink, libnvfatbin,
   libnvptxcompiler, cuda_nvtx, cuda_cuxxfilt, libcurand, libcusparse, libcusolver` e juntamos tudo numa pasta (3,9 GB — um
   superconjunto; não procuramos o mínimo). Atenção: `cccl` e `libnvvm` têm chaves diferentes das que você imaginaria no manifesto.
3. **Compilar** (≈ 20–25 min, ~1200 etapas). O script do repositório precisa de bypass da política de execução do PowerShell:

   ```powershell
   $env:CUDA_PATH = "D:\cuda\toolkit"; $env:CUDAToolkit_ROOT = $env:CUDA_PATH
   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_windows.ps1 `
       -Preset windows-cuda-release -Target audiocpp_server -CudaArchitectures "120a-real"
   ```
   `120a-real` serve só para placas Blackwell (RTX 50xx): acelera a compilação, mas o binário não roda em outras GPUs.
4. O resultado é **um único `audiocpp_server.exe`**, sem DLLs. Copie as DLLs (`ggml*.dll`, `cublas*.dll`, `cudart*.dll`…) da pasta `bin` do
   zip do release para junto dele. `--version` deve mostrar o seu build, e a config de streaming do Fish deve passar a iniciar.

**Arquivos GGUF:** o Q8_0 vem do repositório de GGUFs do audio.cpp (`audio-cpp/audio.cpp-gguf`); Q6_K e Q4_K são gerados com a receita do próprio
projeto — leia `docs/proposals/fish_audio_quantization_recipe.md` na árvore de código. O ponto-chave: os tensores pequenos do "decodificador
acústico rápido" precisam ficar na precisão original (`--keep-type 'model_weights/fast_*=orig'`) e o arquivo de metadados dos tokens deve ser
extraído do Q8; quantizar "no ingênuo" gera áudio embaralhado. Tamanhos: Q8 6,3 GB, Q6 5,4 GB, Q4 4,5 GB.

## 3. Config do servidor e pedido

`server.json` (clonagem de voz = um preset com wav de referência **e a transcrição exata dele**):

```json
{ "host": "127.0.0.1", "port": 18190, "backend": "cuda", "device": 0, "busy_timeout_ms": 600000,
  "models": [{ "id": "fish", "family": "fish_audio", "task": "tts", "mode": "streaming",
               "path": "D:/models/Fish-Audio-S2-Pro-GGUF/fish-audio-s2-pro-q4_k.gguf",
               "model_spec_override": "D:/audiocpp/src/model_specs",
               "session_options": { "fish_audio.reference_cache_slots": "4" },
               "voice_presets": { "minha_voz": { "voice_ref": "D:/voices/ref.wav", "reference_text": "transcrição exata de ref.wav" } } }] }
```
(`model_spec_override` é o que usamos; não testamos se um binário novo o dispensa.)

Pedido: `POST /v1/audio/speech` com `{"model":"fish","input":"...","voice":"minha_voz","response_format":"pcm","stream_format":"sse"}`.
Vêm eventos SSE; `speech.audio.delta` traz **PCM16 mono em base64 a 44,1 kHz**. Fechar a conexão HTTP interrompe a geração
(medido: depois de uma interrupção, o pedido seguinte voltou a falar em 0,65 s).

Nosso adaptador ([`fish_tts_adapter.py`](fish_tts_adapter.py), ~200 linhas, biblioteca padrão + numpy/scipy/requests) transforma isso em
"POST do texto → PCM16 cru a 24 kHz conforme é gerado", e ainda: liga/desliga o audio.cpp, `/sleep` + `/wake` para liberar VRAM, e um *Job Object* do Windows
para o audio.cpp morrer junto com o adaptador (um servidor órfão segura ~5 GB de VRAM). Testado: primeiro byte em 0,5 s.
Dois detalhes que valem copiar: conversão 44,1→24 kHz em streaming com `resample_poly` **em blocos com contexto dos dois lados** (idêntica bit a bit
à conversão de uma vez, verificada com vários tamanhos de bloco — sem emendas) e um limitador suave em vez de corte seco.

## 4. Resultados

### 4.1 Só o TTS (GPU livre), streaming
38 falas por GGUF (2 vozes × 19 textos, de 2 a 34 s de fala). "Geração ÷ áudio" > 1 significa mais lento que o tempo real.

| GGUF | primeiro áudio | tempo de geração ÷ tempo do áudio (faixa, mediana) | pico de VRAM com ~2 GB de desktop |
|---|---|---|---|
| Q8_0 | 0,15–0,31 s | 0,98–1,27, mediana 1,12 | 8,4 GB |
| Q6_K | 0,17–0,35 s | 0,98–1,19, mediana 1,09 | 7,7 GB |
| **Q4_K** | 0,14–0,31 s | 0,85–1,08, **mediana 1,00** | **7,0 GB** |

Atraso inicial necessário para a reprodução nunca secar, em respostas de ~19 s (medido pelos tempos de chegada do SSE):
Q8 1,0–2,5 s, Q4 0,2–0,5 s. Ou seja, **o Q4 não é só o menor: é o que toca mais redondo.** No ouvido (espanhol, um único ouvinte) os três
soaram igualmente bons; o ouvinte escolheu o Q4 por eficiência.
O tom emocional saiu do próprio texto (pontuação, palavras); **não** precisamos de marcas de emoção inline e não as avaliamos.

### 4.2 Pipeline completo (LLM real + TTS real, placa de som simulada que conta falhas)
Q4, respostas curtas de conversa (~8–10 s de fala cada). "Colchão" = áudio acumulado antes de começar a tocar; "fila" = quanto a geração pode adiantar.

| colchão / fila | falhas audíveis | fala testada |
|---|---|---|
| 0,5 s / 6 | **1** | 48 s |
| 0,65 s / 60 | 0 | 52 s |
| **0,8 s / 60** (nossa config) | 0 / 0 / 0 | 50 + 51 + 48 s |
| 1,1 s / 60 | 0 | 52 s |

Com o Q6 e o colchão pequeno, uma sessão real registrou **8 falhas numa única resposta de 11 s** — são os "estalinhos" que o ouvinte relatou.
Tempo do pedido até o primeiro som: **~1,9–2,4 s de mediana** nessas rodadas (≈0,7 s disso é o LLM escrevendo a primeira frase).

### 4.3 VRAM
`gemma4:12b` (≈8,5 GB dedicados) + Fish Q4 + desktop do Windows: **~14,1 GB dos 16 GB** (eram ~15,7 GB com o TTS que substituímos).

## 5. Bônus: fazer o LLM começar a responder mais cedo (Ollama / llama.cpp, modelos de janela deslizante)

Antes: cada resposta relia o prompt inteiro de ~5 000 tokens (**3,4 s**) embora só o último turno tivesse mudado. No `gemma4` (atenção de janela
deslizante) o llama.cpp só consegue reaproveitar um prefixo em cache por meio de *context checkpoints*, e só os cria no **fim de um pedido** — ou seja,
depois da parte que muda a cada turno, então nunca serviam. (O log do servidor mostra `forcing full prompt re-processing due to lack of cache data`.)

O que funcionou (Ollama 0.35.1, sem variável de ambiente): depois de cada resposta, quando a fala termina, mandar **um pedido extra só com o prefixo
estável** (prompt de sistema + histórico, *sem* o bloco volátil de "contexto atual" e *sem* a nova mensagem do usuário) com `num_predict: 1`.
O checkpoint passa a cair exatamente onde o prompt do próximo turno diverge.
Mantenha ferramentas, `num_ctx`, `num_batch` e `keep_alive` idênticos aos pedidos reais, ou o modelo recarrega. Coloque tudo que muda a cada turno
(relógio, memórias recuperadas) **depois** do histórico.

* Medido numa conversa de 8 turnos reproduzida: releitura antes de responder **3,4 s → ~1,2 s** (o aquecimento leva ~0,3–0,7 s, com o assistente parado); no uso real 1,3–1,7 s.
* O que **não** ajudou: `LLAMA_ARG_CHECKPOINT_MIN_SPACING_NT=256` (sem efeito); `LLAMA_ARG_SWA_FULL=1` funciona, mas custa ~2,2 GB de VRAM.
* Pôr o perfil do usuário *antes* do histórico pouparia mais ~0,6 s, mas com este 12B piorou a lembrança da *conversa atual*
  (perguntado "como te disse que ela se chamava?", respondia pelo perfil: 0/6 certas contra 6/6). Teste esse tipo de regressão se reordenar prompts.
* `OLLAMA_DEBUG=1` deixou a leitura do prompt ~7× mais lenta nas nossas rodadas (23 s contra 3 s); não deixe ligado ao medir.
* Não testado: Ollama mais novo (0.40.x) — as notas dele eram sobretudo de Apple/MLX.

## 6. Armadilhas, em resumo
* Não confie em "streaming suportado" no README: procure no release, ou teste uma config com `"mode":"streaming"`.
* Streaming dá *tempo até o primeiro áudio*, não velocidade. Se a geração é ≥ 1,0× a duração do áudio, ainda precisa de colchão.
* Mate processos `audiocpp_server.exe` órfãos — cada um segura ~5 GB de VRAM.
* A primeira subida depois do boot foi lenta (≈57 s, disco frio); depois 10–20 s. Acordar um adaptador que pôs o audio.cpp para dormir levou ≈10 s.
* Se a política do PowerShell bloquear scripts, use `-ExecutionPolicy Bypass` só naquele comando.
* No Git Bash, barras invertidas em heredocs são comidas (corrompemos um caminho do Windows num documento assim).

## 7. Não medido / limites
Outros idiomas; outras vozes; taxa de erro de palavras ou qualquer métrica objetiva de qualidade (só um ouvinte humano); comportamento jogando ou renderizando ao mesmo tempo;
sessões longas (horas); versões do audio.cpp mais novas que `b0feeff`; GPUs que não sejam Blackwell.

## 8. Nota de licença
O cartão do modelo Fish Audio S2 Pro restringe os pesos a uso **não comercial** — confira os termos atuais antes de construir algo além de uso pessoal.
audio.cpp, llama.cpp e Ollama têm licenças próprias.

## Licenças
Código: **MIT** · texto e dados: **CC BY 4.0** (reuso com crédito, uso comercial permitido) · amostras de áudio e a mascote Ali: **CC BY-NC 4.0** (não comercial). Detalhes em [LICENSES.md](LICENSES.md).

## Créditos
Isto foi um **trabalho de equipe**:

* **[@andreaabcj-cmd](https://github.com/andreaabcj-cmd)** — direção do produto, o hardware, todas as decisões e os testes de escuta (os únicos ouvidos humanos por trás dos "soa bem").
* **Claude** (Anthropic, via Claude Code) — engenharia, medições, pesquisa e este texto. Todos os números vêm de scripts rodados na máquina acima; os dados brutos estão em [`data/`](data/).
* **A comunidade open source**, cujo trabalho tornou isto possível e cujos achados lemos antes de medir qualquer coisa ([RESEARCH.md](RESEARCH.md)): o projeto
  [audio.cpp](https://github.com/0xShug0/audio.cpp) e a receita de quantização do Fish, a [Fish Audio](https://fish.audio) e as issues do `fish-speech`,
  o [s2.cpp](https://github.com/rodrigomatta/s2.cpp) e quem publicou GGUFs no Hugging Face, o SGLang-Omni, o llama.cpp (e os mantenedores que documentaram o comportamento dos checkpoints de janela deslizante), o Ollama, e todos que postaram benchmarks e relatos de erro.

Medições brutas: [`data/`](data/) · O que lemos antes: [RESEARCH.md](RESEARCH.md) · Adaptador de exemplo: [`fish_tts_adapter.py`](fish_tts_adapter.py)
