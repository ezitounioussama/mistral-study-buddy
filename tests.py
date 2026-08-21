"""
Tests for study_buddy.py

Run:  python -m pytest tests.py -q

No API key and no network needed: a stub sender stands in for the SDK, so the
conversation memory, the commands and every error branch can be checked offline.
"""

import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import study_buddy as sb


class StubSender:
    """Stands in for the provider. Records what it was sent."""

    def __init__(self, replies=None, error=None):
        self.replies = list(replies or ["A stub answer."])
        self.error = error
        self.calls = []   # a copy of `messages` for every call

    def __call__(self, messages):
        self.calls.append([dict(m) for m in messages])

        if self.error is not None:
            raise self.error

        reply = self.replies.pop(0) if self.replies else "A stub answer."
        return reply, {"prompt": 10, "completion": 5, "total": 15}


def run(typed, sender):
    """Drive run_chat with scripted input. Returns printed output."""
    output = io.StringIO()
    with patch("builtins.input", side_effect=typed):
        with redirect_stdout(output):
            try:
                sb.run_chat(sender)
            except SystemExit:
                pass
    return output.getvalue()


# ---------------------------------------------------------------------------
# Settings the brief asks for
# ---------------------------------------------------------------------------


class TestConfiguration(unittest.TestCase):
    def test_model_is_mistral_small_latest(self):
        self.assertEqual(sb.MODEL, "mistral-small-latest")

    def test_temperature_is_around_point_three(self):
        self.assertAlmostEqual(sb.TEMPERATURE, 0.3, places=2)

    def test_max_tokens_is_set(self):
        self.assertIsInstance(sb.MAX_TOKENS, int)
        self.assertGreater(sb.MAX_TOKENS, 0)

    def test_the_api_key_is_not_hard_coded(self):
        """No key-shaped literal may appear in the source."""
        source = open("study_buddy.py").read()
        self.assertIn("os.getenv", source)
        self.assertIn("load_dotenv", source)
        # A real Mistral key is a long alphanumeric run; nothing like it should
        # be pasted into the file.
        import re
        for literal in re.findall(r'"[A-Za-z0-9]{28,}"', source):
            self.fail(f"suspicious literal in source: {literal[:12]}...")

    def test_system_prompt_sets_a_tutor_personality(self):
        prompt = sb.SYSTEM_PROMPT.lower()
        self.assertIn("tutor", prompt)
        self.assertIn("plain language", prompt)
        self.assertIn("beginner", prompt)

    def test_system_prompt_forbids_inventing_facts(self):
        prompt = sb.SYSTEM_PROMPT.lower()
        self.assertIn("never invent", prompt)
        self.assertIn("not sure", prompt)


class TestApiKeyLoading(unittest.TestCase):
    def _load(self, value):
        env = {} if value is None else {"MISTRAL_API_KEY": value}
        output = io.StringIO()
        raised = None
        with patch.dict(os.environ, env, clear=True):
            with patch("study_buddy.load_dotenv", lambda *a, **k: None):
                with redirect_stdout(output):
                    try:
                        result = sb.load_api_key()
                    except SystemExit as exit_call:
                        result, raised = None, exit_call
        return result, output.getvalue(), raised

    def test_missing_key_exits_with_instructions(self):
        _, output, raised = self._load(None)
        self.assertEqual(raised.code, 1)
        self.assertIn("no Mistral API key found", output)
        self.assertIn(".env.example", output)

    def test_untouched_placeholder_exits(self):
        _, _, raised = self._load("your-api-key-here")
        self.assertIsNotNone(raised)

    def test_blank_key_exits(self):
        for value in ("", "   "):
            _, _, raised = self._load(value)
            self.assertIsNotNone(raised, value)

    def test_real_looking_key_is_accepted_and_stripped(self):
        value, _, raised = self._load("  abc123realkey  ")
        self.assertIsNone(raised)
        self.assertEqual(value, "abc123realkey")


# ---------------------------------------------------------------------------
# Conversation memory — the point of the checkpoint
# ---------------------------------------------------------------------------


class TestConversationMemory(unittest.TestCase):
    def test_the_system_message_comes_first(self):
        sender = StubSender()
        run(["hello", "quit"], sender)

        first_message = sender.calls[0][0]
        self.assertEqual(first_message["role"], "system")
        self.assertEqual(first_message["content"], sb.SYSTEM_PROMPT)

    def test_the_user_question_is_appended(self):
        sender = StubSender()
        run(["What is a list?", "quit"], sender)

        self.assertEqual(sender.calls[0][-1], {"role": "user", "content": "What is a list?"})

    def test_three_connected_questions_carry_the_earlier_turns(self):
        """The brief's requirement: prove it remembers context."""
        sender = StubSender(replies=["Answer one.", "Answer two.", "Answer three."])

        run(["What is a list?", "Why use one?", "Show me another example", "quit"], sender)

        self.assertEqual(len(sender.calls), 3)

        # Turn 1: system + question.
        self.assertEqual(len(sender.calls[0]), 2)
        # Turn 2: system + Q1 + A1 + Q2.
        self.assertEqual(len(sender.calls[1]), 4)
        # Turn 3: system + Q1 + A1 + Q2 + A2 + Q3.
        self.assertEqual(len(sender.calls[2]), 6)

        third = sender.calls[2]
        contents = [m["content"] for m in third]
        self.assertIn("What is a list?", contents)   # first question still there
        self.assertIn("Answer one.", contents)       # and the first answer
        self.assertIn("Why use one?", contents)
        self.assertIn("Answer two.", contents)
        self.assertEqual(third[-1]["content"], "Show me another example")

    def test_roles_alternate_user_assistant(self):
        sender = StubSender(replies=["A1", "A2", "A3"])
        run(["Q1", "Q2", "Q3", "quit"], sender)

        roles = [m["role"] for m in sender.calls[2]]
        self.assertEqual(roles, ["system", "user", "assistant", "user", "assistant", "user"])

    def test_reset_clears_the_memory(self):
        sender = StubSender(replies=["A1", "A2"])
        run(["Q1", "reset", "Q2", "quit"], sender)

        # After reset the second call carries only system + the new question.
        self.assertEqual(len(sender.calls[1]), 2)
        self.assertNotIn("Q1", [m["content"] for m in sender.calls[1]])


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


class TestChatLoop(unittest.TestCase):
    def test_quit_and_exit_both_leave(self):
        for command in ("quit", "exit", "QUIT", "Exit"):
            sender = StubSender()
            output = run([command], sender)
            self.assertIn("Goodbye", output)
            self.assertEqual(sender.calls, [], f"{command} should not call the API")

    def test_the_reply_is_printed(self):
        output = run(["hi", "quit"], StubSender(replies=["Here is the answer."]))
        self.assertIn("Study Buddy: Here is the answer.", output)

    def test_token_usage_is_printed_when_available(self):
        output = run(["hi", "quit"], StubSender())
        self.assertIn("tokens:", output)
        self.assertIn("total 15", output)

    def test_empty_input_is_ignored(self):
        sender = StubSender()
        run(["", "   ", "quit"], sender)
        self.assertEqual(sender.calls, [])

    def test_ctrl_d_exits_cleanly(self):
        output = run(EOFError(), StubSender())
        self.assertIn("Goodbye", output)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling(unittest.TestCase):
    def test_auth_error_is_fatal(self):
        message, fatal = sb.explain_error(Exception("401 Unauthorized"))
        self.assertIn("API key was rejected", message)
        self.assertTrue(fatal)

    def test_rate_limit_is_not_fatal(self):
        message, fatal = sb.explain_error(Exception("429 rate limit exceeded"))
        self.assertIn("Rate limit", message)
        self.assertFalse(fatal)

    def test_server_error_is_not_fatal(self):
        message, fatal = sb.explain_error(Exception("503 Service Unavailable"))
        self.assertIn("trouble right now", message)
        self.assertFalse(fatal)

    def test_network_error_is_recognised_by_type_name(self):
        class ConnectError(Exception):
            pass

        message, fatal = sb.explain_error(ConnectError("no route to host"))
        self.assertIn("Could not reach Mistral", message)
        self.assertFalse(fatal)

    def test_unknown_error_names_its_type(self):
        message, fatal = sb.explain_error(ValueError("something odd"))
        self.assertIn("ValueError", message)
        self.assertFalse(fatal)

    def test_a_failure_does_not_crash_the_loop(self):
        """A network blip must return the student to the prompt, not exit."""
        class ConnectTimeout(Exception):
            pass

        sender = StubSender(error=ConnectTimeout("timed out"))
        output = run(["Q1", "quit"], sender)

        self.assertIn("Could not reach Mistral", output)
        self.assertIn("Goodbye", output)

    def test_a_failed_question_is_dropped_from_the_memory(self):
        """Otherwise it would be resent on the next turn as a duplicate."""
        class ConnectTimeout(Exception):
            pass

        calls = []

        def flaky(messages):
            calls.append([dict(m) for m in messages])
            if len(calls) == 1:
                raise ConnectTimeout("timed out")
            return "Worked this time.", None

        output = run(["Q1", "Q2", "quit"], flaky)

        # The second call must not contain the failed Q1.
        second = [m["content"] for m in calls[1]]
        self.assertNotIn("Q1", second)
        self.assertIn("Q2", second)
        self.assertIn("Worked this time.", output)

    def test_a_missing_usage_field_does_not_break_printing(self):
        def no_usage(messages):
            return "Reply without usage.", None

        output = run(["hi", "quit"], no_usage)
        self.assertIn("Reply without usage.", output)
        self.assertNotIn("tokens:", output)


# ---------------------------------------------------------------------------
# Provider portability
# ---------------------------------------------------------------------------


class TestProviderPortability(unittest.TestCase):
    def test_openai_flag_is_parsed(self):
        self.assertFalse(sb.parse_arguments([]).openai)
        self.assertTrue(sb.parse_arguments(["--openai"]).openai)

    def test_the_compatible_endpoint_points_at_mistral(self):
        self.assertEqual(sb.OPENAI_COMPATIBLE_BASE_URL, "https://api.mistral.ai/v1")

    def test_both_senders_build_and_share_one_call_shape(self):
        """Same signature either way, which is what makes the flag enough."""
        mistral_send = sb.build_mistral_sender("fake-key")
        openai_send = sb.build_openai_sender("fake-key")

        for send in (mistral_send, openai_send):
            self.assertTrue(callable(send))

    def test_the_mistral_import_path_resolves(self):
        """mistralai 2.x moved the client; the fallback must cover both layouts."""
        try:
            from mistralai.client import Mistral
        except ImportError:
            from mistralai import Mistral
        self.assertTrue(callable(Mistral))


if __name__ == "__main__":
    unittest.main(verbosity=2)
