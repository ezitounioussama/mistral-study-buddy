# Notes — setup, settings, memory, errors, and the SDK import

| File | What it is |
|---|---|
| `study_buddy.py` | The assistant |
| `.env.example` | Template for the key — copy to `.env` |
| `.gitignore` | Contains `.env`, so the key cannot be committed |
| `requirements.txt` | `mistralai`, `python-dotenv` (+ `openai` for the stretch goal) |
| `tests.py` | 32 tests, no API key needed |
| `docs/transcript.txt` | Multi-turn session showing the memory working |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
```

Open `.env` and paste your key:

```
MISTRAL_API_KEY=your-real-key-here
```

Get one from https://console.mistral.ai/api-keys/. The key lives only in `.env`, which is listed in
`.gitignore` — it never appears in the Python source. Run without a key and the program stops with
setup instructions rather than a confusing 401.

```bash
python study_buddy.py            # mistralai SDK
python study_buddy.py --openai   # same model via the OpenAI-compatible endpoint
python -m pytest tests.py -q     # 32 passed
```

Type a question and press Enter. `quit` or `exit` to leave, `reset` to clear the conversation.
`Ctrl+C` and `Ctrl+D` also exit cleanly.

## Settings

| Setting | Value | Why |
|---|---|---|
| Model | `mistral-small-latest` | Required by the checkpoint; small and cheap suits Q&A |
| Temperature | `0.3` | Low, so the same question gets the same explanation. Not `0`, which reads stiff |
| Max tokens | `600` | Caps reply length — a tutor answer running for pages is worse than a short one, and tokens are billed |

## How the memory works

`messages` starts with the system prompt and grows by two entries per turn:

```python
messages = [{"role": "system", "content": SYSTEM_PROMPT}]
...
messages.append({"role": "user", "content": question})     # before the call
reply, usage = send(messages)
messages.append({"role": "assistant", "content": reply})   # after the call
```

The whole list is sent every time. That is the only reason follow-ups work: **the model itself
remembers nothing between calls**, so context is something the program supplies, not something the
API stores.

See [`docs/transcript.txt`](docs/transcript.txt) for a four-turn session. Turn 2 and 3 are
follow-ups that only make sense with the earlier turns present, and turn 4 asks the assistant to
recall the first question.

## Error handling

Every call is wrapped, and errors are split into fatal and recoverable:

| Situation | Behaviour |
|---|---|
| No key, or `.env.example` copied but not edited | Setup instructions, exit 1 |
| Key rejected (401) | Explains and exits — a bad key will not fix itself |
| Rate limit (429) | Says to wait, **keeps running** |
| Server error (5xx) | Says to retry, keeps running |
| No internet / timeout | Explains, keeps running |
| Anything unexpected | Names the exception type, keeps running |

A failed question is also **removed from `messages`**, so it is not resent as a duplicate on the
next turn. A dropped connection should not cost the student their conversation.

## Stretch goal — provider portability

Both providers are wrapped in one function with the same signature:

```python
send(messages) -> (reply_text, usage_or_None)
```

The chat loop only calls that, so it has no idea which SDK is underneath. `--openai` swaps the
`mistralai` SDK for the `openai` SDK pointed at Mistral:

```python
client = OpenAI(api_key=api_key, base_url="https://api.mistral.ai/v1")
```

Same model, same messages, same key — only `base_url` changes. Switching provider is a flag, not a
rewrite.

## Note on the SDK import path

The checkpoint warns to check the installed version if the import differs from the handbook, and it
does. The installed version here is **mistralai 2.9.3**, which has **no top-level `__init__.py`** —
so the handbook's `from mistralai import Mistral` raises:

```
ImportError: cannot import name 'Mistral' from 'mistralai' (unknown location)
```

In 2.x the client moved to `mistralai.client`. The code tries both layouts:

```python
try:
    from mistralai.client import Mistral   # mistralai 2.x
except ImportError:
    from mistralai import Mistral         # mistralai 1.x
```

A test asserts one of the two resolves, so a future version bump fails loudly instead of at runtime.

## Testing

`python -m pytest tests.py -q` → **32 passed**, no key and no network required. A stub provider
records what it was sent, which is what makes the memory checkable: one test asserts the message
list grows 2 → 4 → 6 across three turns and that the first question and answer are still present in
the third call.

The 401 path was also verified against the live API using a deliberately invalid key — it produced
the friendly message and exit code 1, not a traceback.
