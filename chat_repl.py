"""Interactive REPL for chatbot.py against a small fake in-memory dataset -
no Postgres needed. CHAT_MODE/LITELLM_* still come from .env (via chatbot.py's
own load_dotenv()), so this will genuinely try your real LiteLLM proxy if
CHAT_MODE=llm is set there, falling back to simple mode if it's unreachable.

Run: <venv>/Scripts/python.exe chat_repl.py
Type 'exit' or 'quit' to stop.
"""
import sys
import types
from datetime import datetime, timedelta, timezone

sys.path.insert(0, __file__.rsplit("\\", 1)[0])

# --- fake dataset, stands in for db.postgres ---
_STORES = [
    {"store_id": "store1", "store": "Koramangala", "region": "South"},
    {"store_id": "store2", "store": "HSR Layout", "region": "South"},
    {"store_id": "store3", "store": "Indiranagar", "region": "East"},
]
_FOOTFALL = {"store1": 128, "store2": 74, "store3": 201}
_DEMOGRAPHICS = {
    "store1": {"gender_breakdown": {"female": 61, "male": 67}, "age_group_breakdown": {"18-25": 40, "26-40": 62, "41+": 26}},
    "store2": {"gender_breakdown": {"female": 30, "male": 44}, "age_group_breakdown": {"18-25": 20, "26-40": 35, "41+": 19}},
    "store3": {"gender_breakdown": {"female": 96, "male": 105}, "age_group_breakdown": {"18-25": 70, "26-40": 90, "41+": 41}},
}

db_pkg = types.ModuleType("db")
postgres_mod = types.ModuleType("db.postgres")
postgres_mod.fetch_stores = lambda: _STORES
postgres_mod.get_footfall_count = lambda store_id, start, end: _FOOTFALL.get(store_id, 0)
postgres_mod.get_demographics_breakdown = lambda store_id, start, end: _DEMOGRAPHICS.get(
    store_id, {"gender_breakdown": {}, "age_group_breakdown": {}}
)
postgres_mod.list_entry_exit_logs = lambda store_id, person_id, start, end, limit: [
    {"event_type": "entry", "event_time": datetime.now(timezone.utc) - timedelta(minutes=5), "person_id": "p-001"}
]
postgres_mod.list_persons = lambda store_id, start, end, limit: [{"id": i} for i in range(_FOOTFALL.get(store_id, 0))]
db_pkg.postgres = postgres_mod
sys.modules["db"] = db_pkg
sys.modules["db.postgres"] = postgres_mod
# --- end fake dataset ---

import chatbot  # noqa: E402

print(f"CCTV analytics chatbot (mode={chatbot.CHAT_MODE}) - fake in-memory data, stores: "
      f"{', '.join(s['store_id'] for s in _STORES)}")
print("Try: 'footfall for store1 today', 'gender breakdown for store2', "
      "'how crowded was Indiranagar', 'show me visits yesterday'")
print("Type 'exit' to quit.\n")

conversation = []
while True:
    try:
        user_input = input("you> ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if user_input.lower() in ("exit", "quit"):
        break
    if not user_input:
        continue
    answer, conversation = chatbot.chat(user_input, conversation)
    print(f"bot> {answer}\n")
