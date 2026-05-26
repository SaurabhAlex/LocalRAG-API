# TODO

- [x] Identify root cause of `All connection attempts failed` in `/qa/ask-question`.
- [x] Root cause identified: Ollama at `settings.ollama_base_url` is unreachable.
- [ ] Make `/qa/ask-question` error message include Ollama URL + underlying httpx error.
- [ ] Add optional `/qa/ollama-health` endpoint to test connectivity.
- [ ] Add notes on required env var `OLLAMA_BASE_URL` when running FastAPI in Docker.


