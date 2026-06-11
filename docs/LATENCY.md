# Voice-turn latency: where the time goes

Analysis from a live browser-audio session on the NVIDIA client (2026-06-11),
config: `parakeet_onnx` (CUDA, int8) STT, `kokoro` (CUDA) TTS, `langgraph`
LLM backend against LM Studio (`unsloth/gemma-4-e4b-it`).

Measured turn (dashboard metrics panel):

| stage     | last turn |
|-----------|-----------|
| stt       | 479 ms    |
| llm ttft  | 1579 ms   |
| llm total | 0 ms      |
| tts ttfa  | 335 ms    |

Voice-to-voice is the clean partition `stt + llm ttft + tts ttfa` ≈ **2.4 s**
(see `_DERIVED` in `voice_agent/metrics.py`).

## Why "llm ttft" is much higher than LM Studio's own numbers

The dashboard's `llm ttft` is **not** one LM Studio request. Two effects stack:

1. **Two sequential LLM calls per turn.** The LangGraph turn
   (`voice_agent/backends/llm/langgraph_helmsman/graph.py`) first runs the
   *classify* node (intent: COMMAND vs QUESTION, `max_tokens=8`), then the
   *command* node (the full JSON action response). LM Studio's UI shows
   per-request stats; the dashboard shows the sum.
2. **No streaming.** `LangGraphLLMService` awaits the whole graph and emits
   the result as a single `LLMTextFrame`
   (`voice_agent/backends/llm/langgraph_helmsman/service.py`). So `llm ttft`
   includes the command call's *entire* prompt evaluation **and** generation,
   not just time-to-first-token. This is also why `llm total` reads 0 ms:
   first and last "token" are the same single frame — the metric is
   meaningless for this backend.

There is a sneakier cost on top: the classify and command calls use
**different system prompts**, so on a single llama.cpp slot each call
invalidates the KV prefix cache and the full system prompt is re-evaluated
every single turn.

## Options to lower it (ranked by payoff)

1. **Fold classify into the command call.** Add a `question` action type to
   the command JSON schema and let one call do both jobs. Commands — the
   latency-critical voice path — become a single LLM round trip, and because
   every turn then starts with the same system prompt, llama.cpp's prefix
   cache applies and prompt-eval drops to near zero on repeat turns.
   Expected: roughly half the felt LLM latency, likely more with the cache
   effect. Questions route to the RAG branch after the command call flags
   them (they are slow anyway because of retrieval).
2. **Measure the split first.** Add per-node timings (`classify_ms`,
   `command_ms`, and per RAG node) to the `langgraph_turn` log line so any
   change is judged against a real baseline.
3. **LM Studio knobs:** flash attention on, all layers on GPU, "keep model
   in memory". Small wins if not already set. Also note the model must be
   loaded with a context length large enough for the RAG prompts — the
   rerank step packs `retrieval_top_k` (default 20) chunks into one prompt
   (~3.4k tokens observed); 2560 ctx produced LM Studio 400 errors
   (`n_keep >= n_ctx`), 8192+ is comfortable.
4. **Streaming + incremental JSON parse** so TTS starts before the response
   completes — the biggest possible win for spoken replies, but the action
   JSON must be parsed before the spoken `response` field is known, so it
   needs field-order-aware partial parsing. Deferred.

## Fixing the metric itself

`llm_total_ms` should either be dropped for non-streaming backends or the
service should emit start/first/last timestamps around the graph run so
`llm ttft` vs `llm total` regain their intended meaning (true TTFT vs
generation time). Worth doing together with option 2.
