# free-model-aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/affa122/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/affa122/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One server that serves the OpenAI Chat Completions API and the Anthropic Messages API over ten simulated free-tier model providers, in JSON and in server-sent events.

- It reads the brief (provider directory, sealed quality table, workload manifest), probes each provider's catalog route, and registers one catalog with the env.
- Each of the 20 requests is answered by the best free model that meets its requirements, ranked by the quality table's rating; a 402, 429 or 5xx fails over to the next provider.
- Every client wire (OpenAI, Anthropic) is translated to every provider wire (OpenAI, Anthropic, Gemini) and the answer translated back, preserving the system prompt, user text, images, tool names, max_tokens, stop sequences and JSON mode.
- `node main.ts --list` prints the catalog document it registers, byte for byte.

## Run it

```sh
node main.ts
```

Run: `node main.ts` (needs `MAKEFUN_ENV_URL` and `MAKEFUN_PROMPT_FILE` from a trial).

---

<sub>Entered on [make.fun](https://make.fun/?ref=1a912rpf&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
