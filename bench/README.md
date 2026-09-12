# bench

All scripts talk to the OpenAI-compatible endpoint. Env: `BASE` (default `http://127.0.0.1:8000`),
`API_KEY` (or `/root/.api_key`), `MODEL` (default `glm-5.3-flash`).

| script | what it measures |
|---|---|
| `gates.py` | 9 functional gates: models, long pt-BR generation, thinking on/off form, tool call, vision ×3, prefix cache report, 8-stream concurrency |
| `prose_speed.py` | batch-1 pt-BR prose, 8 × 600 tokens, thinking off; run twice, keep the warm one |
| `code_speed.py` | batch-1 code, 4 × 700 tokens |
| `concurrency.py N TOKENS` | N distinct prompts in parallel, aggregate and per-stream tok/s, two passes |
| `ttft.py` | time to first content token for 1k–27k-token cold prompts (unique salt); `TTFT_SOURCE` = any long text file |
| `per_batch.sh LOG` | decode tok/s and acceptance per running-batch size from the server log |

Run on an idle server; concurrent traffic changes the numerics enough to move every figure.
