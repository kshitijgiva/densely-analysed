"""Live smoke test of chatbot._llm_chat against the real LiteLLM proxy.
db.postgres is stubbed (no Postgres available here) so tool calls return
canned data instead of hitting a real database - this only exercises the
actual model round-trip / tool-calling loop, which is what we fixed."""
import sys
import types
from unittest.mock import MagicMock

sys.path.insert(0, r"C:\Users\Sujith\AppData\Local\Temp\claude\c--Users-Sujith-OneDrive-Desktop-densely-analysed-main-densely-analysed-main\56e2df01-d75a-477e-8aba-9b4ccadcf0e2\scratchpad\chatbot-fix")

db_pkg = types.ModuleType("db")
postgres_mod = types.ModuleType("db.postgres")
postgres_mod.fetch_stores = MagicMock(
    return_value=[{"store_id": "store1", "store": "Main St", "region": "West"}]
)
postgres_mod.get_demographics_breakdown = MagicMock(
    return_value={"gender_breakdown": {"female": 31, "male": 24}, "age_group_breakdown": {"18-25": 20, "26-40": 35}}
)
postgres_mod.get_footfall_count = MagicMock(return_value=57)
postgres_mod.list_entry_exit_logs = MagicMock(return_value=[])
postgres_mod.list_persons = MagicMock(return_value=[])
db_pkg.postgres = postgres_mod
sys.modules["db"] = db_pkg
sys.modules["db.postgres"] = postgres_mod

import chatbot  # noqa: E402

print(f"CHAT_MODE={chatbot.CHAT_MODE}")
print(f"LITELLM_BASE_URL={chatbot.LITELLM_BASE_URL}")
print(f"LITELLM_MODEL={chatbot.LITELLM_MODEL}")
print(f"key set: {bool(chatbot.LITELLM_API_KEY)}, key length: {len(chatbot.LITELLM_API_KEY or '')}")
print()

message = "How many people visited store1 today, and what's the gender breakdown?"
print(f"you> {message}")
answer, history = chatbot.chat(message, [])
print(f"bot> {answer}")
print()
print(f"(stub) get_footfall_count called: {postgres_mod.get_footfall_count.called}")
print(f"(stub) get_demographics_breakdown called: {postgres_mod.get_demographics_breakdown.called}")

if answer.startswith("(LLM unavailable:"):
    print("\n*** FAILED: fell back to simple mode, LLM call did not succeed ***")
    sys.exit(1)
print("\n*** LIVE LLM ROUND-TRIP SUCCEEDED ***")
