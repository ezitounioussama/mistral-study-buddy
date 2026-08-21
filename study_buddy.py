"""
Study Buddy — a command-line study assistant powered by Mistral.

Run:
    python study_buddy.py               # Mistral SDK (default)
    python study_buddy.py --openai      # same model, OpenAI-compatible endpoint
    python study_buddy.py --help

Type 'quit' or 'exit' to leave, 'reset' to clear the conversation.
"""

import argparse
import os
import sys

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = "mistral-small-latest"

# 0.3 is low on purpose. Temperature controls how much randomness is allowed
# when picking each next word: high values invent, low values stay close to the
# most likely wording. A study assistant should give the same student the same
# explanation twice, so it sits near the bottom of the range without being 0
# (which can make answers stiff and repetitive).
TEMPERATURE = 0.3

# A ceiling on the reply length. Two reasons: a tutor answer that runs for pages
# is worse than a short one, and tokens are billed, so an unbounded reply is an
# unbounded cost.
MAX_TOKENS = 600

# The system message. It sets the assistant's behaviour for the whole
# conversation and the student never sees it. The last rule matters most: a tutor
# that invents an answer is worse than one that admits a gap, because the student
# has no way to tell the difference.
SYSTEM_PROMPT = """You are Study Buddy, a patient tutor for a student who is learning to code.

How to answer:
- Explain in plain language. Assume the student is a beginner and define any
  term you introduce.
- Keep answers short by default: a few sentences or a short list. Expand only
  when the student asks for more.
- Use a small, concrete code example when it makes the idea clearer.
- Be encouraging and never condescending. A confused question is a reasonable
  question.
- When the student asks a follow-up such as "why?" or "show me another",
  continue from what you already told them instead of starting over.
- If you are not sure about something, say so plainly. Never invent function
  names, version numbers, or facts. Guessing is worse than admitting the gap,
  because the student cannot tell the two apart.
"""

# The OpenAI-compatible endpoint Mistral exposes, used by --openai.
OPENAI_COMPATIBLE_BASE_URL = "https://api.mistral.ai/v1"

EXIT_COMMANDS = {"quit", "exit"}


# ---------------------------------------------------------------------------
# API key
# ---------------------------------------------------------------------------


def load_api_key() -> str:
    """Read MISTRAL_API_KEY from .env and stop with a clear message if absent.

    The key is read from the environment, never written in this file. load_dotenv
    reads the .env file next to the script and puts its values into the
    environment so os.getenv can see them.

    A leftover placeholder counts as missing: failing here with instructions
    beats sending 'your-api-key-here' and getting an opaque 401 back.
    """
    load_dotenv()

    api_key = os.getenv("MISTRAL_API_KEY", "").strip()
    placeholders = {"your-api-key-here", "your_mistral_api_key", "changeme", "..."}

    if not api_key or api_key in placeholders:
        print("Error: no Mistral API key found.")
        print()
        print("Set it up in three steps:")
        print("  1. cp .env.example .env")
        print("  2. Put your real key after MISTRAL_API_KEY= in .env")
        print("  3. Get a key from https://console.mistral.ai/api-keys/")
        print()
        print(".env is listed in .gitignore, so the key cannot be committed.")
        sys.exit(1)

    return api_key


# ---------------------------------------------------------------------------
# Provider portability
#
# Both providers are wrapped in one tiny function with the same signature:
#     send(messages) -> (reply_text, usage_dict_or_None)
#
# The chat loop only ever calls that, so it has no idea which SDK is underneath.
# Swapping provider is a command-line flag, not a rewrite.
# ---------------------------------------------------------------------------


def build_mistral_sender(api_key: str):
    """Return a send() backed by the official mistralai SDK."""
    # The import path is version-dependent. In mistralai 2.x the client lives at
    # mistralai.client; older 1.x releases exported it from the package root, and
    # 0.x used a differently named MistralClient class. Both modern paths are
    # tried so this works across versions.
    try:
        from mistralai.client import Mistral  # mistralai 2.x
    except ImportError:
        from mistralai import Mistral  # mistralai 1.x

    client = Mistral(api_key=api_key)

    def send(messages):
        response = client.chat.complete(
            model=MODEL,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        reply = response.choices[0].message.content
        return reply, describe_usage(response)

    return send


def build_openai_sender(api_key: str):
    """Return a send() that reaches Mistral through its OpenAI-compatible endpoint.

    This is the stretch goal, and it is the clearest demonstration of provider
    portability: same model, same messages, same key — a different SDK and base
    URL. Only base_url changes to point at Mistral instead of OpenAI.
    """
    try:
        from openai import OpenAI
    except ImportError:
        print("Error: --openai needs the openai package.")
        print("Install it with:  pip install openai")
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url=OPENAI_COMPATIBLE_BASE_URL)

    def send(messages):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        reply = response.choices[0].message.content
        return reply, describe_usage(response)

    return send


def describe_usage(response):
    """Pull token counts off a response, or None when they are absent.

    Both SDKs expose usage with the same field names, but neither guarantees it
    is present, so every field is read with getattr rather than assumed.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return None

    return {
        "prompt": getattr(usage, "prompt_tokens", None),
        "completion": getattr(usage, "completion_tokens", None),
        "total": getattr(usage, "total_tokens", None),
    }


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def explain_error(error) -> tuple[str, bool]:
    """Turn an exception into (message, fatal).

    Fatal errors end the program; the rest return the student to the prompt with
    their conversation intact, because a dropped connection should not throw away
    the whole session.

    Checks run from most specific to least: every SDK error inherits from a
    common base, so a broad check placed first would swallow the specific ones.
    """
    name = type(error).__name__
    text = str(error)
    lowered = text.lower()

    # Authentication — a bad key will not fix itself, so stop.
    if "401" in text or "unauthorized" in lowered or "invalid api key" in lowered:
        return (
            "The API key was rejected.\n"
            "  Check MISTRAL_API_KEY in .env for typos or extra spaces, and that\n"
            "  the key is still active at console.mistral.ai/api-keys/",
            True,
        )

    # Rate limit or quota — temporary.
    if "429" in text or "rate limit" in lowered or "quota" in lowered:
        return (
            "Rate limit reached. Wait a few seconds and ask again.\n"
            "  If it keeps happening, check your usage at console.mistral.ai",
            False,
        )

    # Server-side problem — temporary.
    if any(code in text for code in ("500", "502", "503", "504")):
        return ("Mistral is having trouble right now. Try again in a moment.", False)

    # No network. Checked by name because the exception comes from httpx, which
    # is an internal dependency of the SDK rather than something this file imports.
    if "Connect" in name or "Timeout" in name or "Network" in name:
        return (
            "Could not reach Mistral — this usually means no internet connection.",
            False,
        )

    if "NoResponseError" in name:
        return ("Mistral accepted the request but sent nothing back. Try again.", False)

    # Anything unforeseen. The type is shown because it is useful; the raw text is
    # kept short so internal details do not spill into the terminal.
    return (f"Something went wrong ({name}). Try again, or type quit to leave.", False)


# ---------------------------------------------------------------------------
# Chat loop
# ---------------------------------------------------------------------------


def print_banner(provider: str) -> None:
    print("=" * 62)
    print("                      STUDY BUDDY")
    print("=" * 62)
    print(f"  Model       : {MODEL}")
    print(f"  Provider    : {provider}")
    print(f"  Temperature : {TEMPERATURE}   (low, so answers stay consistent)")
    print(f"  Max tokens  : {MAX_TOKENS}")
    print("  Commands    : 'quit' or 'exit' to leave, 'reset' to start over")
    print("-" * 62)


def run_chat(send, provider: str = "mistral") -> None:
    """Ask, answer, remember — until the student leaves.

    `messages` is the memory. It starts with the system prompt, then grows by two
    entries per turn: the student's question and the assistant's reply. The whole
    list is sent every time, which is what lets a follow-up like "why?" make
    sense — the model itself remembers nothing between calls.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print_banner(provider)

    while True:
        # Ctrl+C and Ctrl+D should leave quietly rather than print a traceback.
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye — keep going!")
            return

        if not question:
            continue

        if question.lower() in EXIT_COMMANDS:
            print("Goodbye — keep going!")
            return

        if question.lower() == "reset":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("Conversation cleared. The assistant has forgotten the earlier turns.")
            continue

        # Append the student's turn BEFORE the call, so it is part of the context.
        messages.append({"role": "user", "content": question})

        try:
            reply, usage = send(messages)

        except Exception as error:  # noqa: BLE001 — sorted out by explain_error
            message, fatal = explain_error(error)
            print(f"\n{message}")

            # Drop the question that failed. Leaving it in would send it again on
            # the next turn and confuse the assistant with a duplicate.
            messages.pop()

            if fatal:
                sys.exit(1)
            continue

        print(f"\nStudy Buddy: {reply}")

        if usage and usage.get("total") is not None:
            print(
                f"\n  [tokens: prompt {usage['prompt']}, "
                f"reply {usage['completion']}, total {usage['total']} "
                f"| turns remembered: {(len(messages) + 1) // 2}]"
            )

        # Append the reply too, so the next turn can build on it.
        messages.append({"role": "assistant", "content": reply})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="A command-line study assistant powered by Mistral."
    )
    parser.add_argument(
        "--openai",
        action="store_true",
        help=(
            "Call the same model through Mistral's OpenAI-compatible endpoint "
            "using the openai SDK, instead of the mistralai SDK."
        ),
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    arguments = parse_arguments(argv)
    api_key = load_api_key()

    if arguments.openai:
        send = build_openai_sender(api_key)
        provider = "openai SDK -> api.mistral.ai/v1 (compatible endpoint)"
    else:
        send = build_mistral_sender(api_key)
        provider = "mistralai SDK"

    run_chat(send, provider)


if __name__ == "__main__":
    main()
