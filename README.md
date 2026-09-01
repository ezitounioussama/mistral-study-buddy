# Study Buddy — a command-line study assistant (Mistral)

A terminal tutor built on the `mistralai` SDK. It keeps the conversation in memory, so follow-ups
like *"why?"* or *"show me another"* work — the model is stateless, so the program is what
supplies the context by resending the whole conversation each turn.

Two things came out of building it. The handbook's import (`from mistralai import Mistral`) does
not work on the installed 2.x SDK, so the code tries both layouts. And a failed question is
removed from the history rather than left in, so a dropped connection doesn't resend a duplicate
next turn.

```bash
uv venv
uv pip install -r requirements.txt
cp .env.example .env          # then paste your MISTRAL_API_KEY

uv run python study_buddy.py            # mistralai SDK
uv run python study_buddy.py --openai   # same model, OpenAI-compatible endpoint
uv run pytest tests.py -q               # 32 passed, no key needed
```

## Also in this repo

- **[NOTES.md](NOTES.md)** — setup, the model settings and why, how the memory works, the error
  table, the provider-portability stretch goal, and the SDK import problem in full
- [`docs/transcript.txt`](docs/transcript.txt) — a four-turn session showing the memory working

---

Author: **Oussama Ezitouni**
