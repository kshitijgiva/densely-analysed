"""Consolidated check of every fix made to chatbot.py this session:
1. _llm_chat actually calls the model and executes tool calls (mocked client)
2. needs_store recognizes "visit" so store_id doesn't leak as None
3. store matching prefers the longer/more specific name
4. chat() caps history length
Run: <venv>/Scripts/python.exe test_all_fixes.py
"""
import os
import sys
import types
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(__file__))

dotenv_mod = types.ModuleType("dotenv")
dotenv_mod.load_dotenv = lambda *a, **k: None
sys.modules["dotenv"] = dotenv_mod

openai_mod = types.ModuleType("openai")
openai_mod.OpenAI = MagicMock()
sys.modules["openai"] = openai_mod

db_pkg = types.ModuleType("db")
postgres_mod = types.ModuleType("db.postgres")
postgres_mod.fetch_stores = MagicMock(
    return_value=[
        {"store_id": "hsr", "store": "HSR"},
        {"store_id": "hsr_layout", "store": "HSR Layout"},
    ]
)
postgres_mod.get_demographics_breakdown = MagicMock(return_value={"gender_breakdown": {}})
postgres_mod.get_footfall_count = MagicMock(return_value=42)
postgres_mod.list_entry_exit_logs = MagicMock(return_value=[])
postgres_mod.list_persons = MagicMock(return_value=[{"id": 1}])
db_pkg.postgres = postgres_mod
sys.modules["db"] = db_pkg
sys.modules["db.postgres"] = postgres_mod

os.environ["CHAT_MODE"] = "simple"
import chatbot  # noqa: E402

failures = []


def check(label, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {label}")
    if not cond:
        failures.append(label)


# 1. LLM tool-calling loop (mocked client, simulate CHAT_MODE=llm directly via _llm_chat)
chatbot.LITELLM_API_KEY = "test-key"


def _tool_call(call_id, name, arguments):
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = arguments
    tc.model_dump.return_value = {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}
    return tc


def _response(tool_calls=None, content=None):
    message = types.SimpleNamespace(tool_calls=tool_calls, content=content)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])


round1 = _response(tool_calls=[_tool_call("c1", "get_footfall", '{"store_id": "hsr"}')])
round2 = _response(content="HSR had 42 visitors.")
fake_client = MagicMock()
fake_client.chat.completions.create.side_effect = [round1, round2]
chatbot._get_client = lambda: fake_client
answer, hist = chatbot._llm_chat("how many people came to hsr?", [])
check("llm tool-calling loop returns model's final answer", answer == "HSR had 42 visitors.")
check("llm tool-calling loop invoked the DB tool", postgres_mod.get_footfall_count.called)

# 2. "visit" keyword no longer leaks store_id=None
postgres_mod.list_persons.reset_mock()
answer, _ = chatbot.chat("show me visits yesterday", [])
check("'visit' resolves a store instead of leaking None", "None" not in answer and ("hsr" in answer))

# 3. store matching prefers longer/more specific name
check("'HSR Layout' resolves to hsr_layout, not hsr", chatbot._resolve_store_id("how busy was HSR Layout today") == "hsr_layout")
check("'HSR' (bare) still resolves to hsr", chatbot._resolve_store_id("how busy was HSR today") == "hsr")

# 4. history is capped
long_history = [{"role": "user" if i % 2 == 0 else "assistant", "content": str(i)} for i in range(50)]
_, new_history = chatbot.chat("footfall for hsr today", long_history)
check(f"history capped at {chatbot.MAX_HISTORY_MESSAGES} (got {len(new_history)})", len(new_history) <= chatbot.MAX_HISTORY_MESSAGES)

print()
if failures:
    print(f"{len(failures)} CHECK(S) FAILED: {failures}")
    sys.exit(1)
print("ALL CHECKS PASSED")
