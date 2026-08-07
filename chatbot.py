"""
Natural-language chat over CCTV analytics data — no external LLM required.

Default mode is a small keyword/intent router that calls the same Postgres
helpers as api.py (footfall, demographics, persons, entry/exit, stores).
Optional LiteLLM tool-calling is available with CHAT_MODE=llm if your proxy
is reachable; it is not needed for the /chat endpoint to work.
"""
import json
import os
import re
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

from db.postgres import (
    fetch_stores,
    get_demographics_breakdown,
    get_footfall_count,
    list_entry_exit_logs,
    list_persons,
)

load_dotenv()

CHAT_MODE = os.environ.get("CHAT_MODE", "simple").lower()  # simple | llm
LITELLM_BASE_URL = os.environ.get("LITELLM_BASE_URL", "https://litellm.internal.givadiva.co")
LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY")
LITELLM_MODEL = os.environ.get("LITELLM_MODEL", "gpt-4o")
MAX_TOOL_ROUNDS = 5

_STORE_RE = re.compile(r"\b(store[_-]?[a-z0-9]+)\b", re.I)
_IDENTIFY_RE = re.compile(
    r"\b(who is|identify|name of|track this person|find this person|facial)\b",
    re.I,
)


def _window_from_text(text: str):
    """Parse a coarse time window from the message; default last 24h."""
    now = datetime.now(timezone.utc)
    lower = text.lower()
    if "this month" in lower or "current month" in lower:
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return start, now
    if "this week" in lower or "current week" in lower:
        start = (now - timedelta(days=now.weekday())).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return start, now
    if "today" in lower:
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return start, now
    if "yesterday" in lower:
        end = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return end - timedelta(days=1), end
    if "last 7 days" in lower or "past week" in lower:
        return now - timedelta(days=7), now
    return now - timedelta(hours=24), now


def _resolve_store_id(text: str):
    match = _STORE_RE.search(text)
    if match:
        return match.group(1)

    stores = fetch_stores() or []
    if len(stores) == 1:
        return stores[0]["store_id"]

    lower = text.lower()
    for store in stores:
        sid = store["store_id"]
        name = (store.get("store") or "").lower()
        if sid.lower() in lower or (name and name in lower):
            return sid
    return None


def _fmt_breakdown(label, mapping):
    if not mapping:
        return f"No {label} data."
    parts = [f"{k}: {v}" for k, v in sorted(mapping.items(), key=lambda kv: (-kv[1], kv[0]))]
    return ", ".join(parts)


def _simple_chat(message: str, history=None):
    history = history or []
    text = (message or "").strip()
    if not text:
        return "Ask about stores, footfall, demographics, persons, or entry/exit logs.", history

    if _IDENTIFY_RE.search(text):
        answer = (
            "I only answer aggregate analytics (footfall, demographics, visit counts). "
            "Identifying or naming individuals is out of scope."
        )
        history = history + [
            {"role": "user", "content": text},
            {"role": "assistant", "content": answer},
        ]
        return answer, history

    lower = text.lower()
    start, end = _window_from_text(text)

    if any(k in lower for k in ("list store", "which store", "what store", "all store", "stores")):
        stores = fetch_stores() or []
        if not stores:
            answer = "No stores are registered yet."
        else:
            lines = [
                f"- {s['store_id']} ({s.get('store') or s['store_id']}, region={s.get('region')})"
                for s in stores
            ]
            answer = "Stores:\n" + "\n".join(lines)
        history = history + [
            {"role": "user", "content": text},
            {"role": "assistant", "content": answer},
        ]
        return answer, history

    store_id = _resolve_store_id(text)
    gender_focus = None
    if re.search(r"\b(females?|women|woman)\b", lower):
        gender_focus = "female"
    elif re.search(r"\b(males?|men|man|boys?)\b", lower):
        gender_focus = "male"

    needs_store = store_id is not None or any(
        k in lower
        for k in (
            "footfall",
            "visitor",
            "visitors",
            "unique",
            "demographic",
            "gender",
            "female",
            "male",
            "women",
            "men",
            "age",
            "person",
            "entry",
            "exit",
            "log",
            "came",
            "how many",
        )
    )
    if needs_store and not store_id:
        stores = fetch_stores() or []
        ids = ", ".join(s["store_id"] for s in stores) or "(none)"
        answer = f"Which store? Known store_ids: {ids}"
        history = history + [
            {"role": "user", "content": text},
            {"role": "assistant", "content": answer},
        ]
        return answer, history

    if gender_focus or any(
        k in lower for k in ("demographic", "gender", "age breakdown", "age group", "age")
    ):
        data = get_demographics_breakdown(store_id, start, end)
        gender = data.get("gender_breakdown", {}) or {}
        if gender_focus:
            count = int(gender.get(gender_focus, 0))
            answer = (
                f"{count} {gender_focus} visitors at {store_id} "
                f"from {start.isoformat()} to {end.isoformat()} "
                f"(full gender split: {_fmt_breakdown('gender', gender)})."
            )
        else:
            answer = (
                f"Demographics for {store_id} from {start.isoformat()} to {end.isoformat()}:\n"
                f"- Gender: {_fmt_breakdown('gender', gender)}\n"
                f"- Age: {_fmt_breakdown('age', data.get('age_group_breakdown', {}))}"
            )
    elif any(
        k in lower
        for k in (
            "footfall",
            "unique visitor",
            "how many visitor",
            "visitor count",
            "how many people",
            "how many came",
            "people came",
        )
    ):
        count = get_footfall_count(store_id, start, end)
        answer = (
            f"Unique footfall for {store_id} from {start.isoformat()} to {end.isoformat()}: "
            f"{count}"
        )
    elif any(k in lower for k in ("entry", "exit", "log")):
        rows = list_entry_exit_logs(store_id, None, start, end, 20) or []
        answer = (
            f"{len(rows)} recent entry/exit events for {store_id} "
            f"(showing up to 20) between {start.isoformat()} and {end.isoformat()}."
        )
        if rows:
            sample = rows[0]
            answer += (
                f" Latest: {sample.get('event_type')} at {sample.get('event_time')} "
                f"(person_id={sample.get('person_id')})."
            )
    elif "person" in lower or "visit" in lower:
        rows = list_persons(store_id, start, end, 20) or []
        answer = (
            f"{len(rows)} visit records for {store_id} in that window "
            f"(showing up to 20). Use the /stores/{store_id}/persons API for full detail."
        )
    else:
        answer = (
            "I can answer: stores, footfall, demographics (gender/age), persons, "
            "or entry/exit logs. Example: \"footfall for store2 today\" or "
            "\"gender breakdown for store2\"."
        )

    history = history + [
        {"role": "user", "content": text},
        {"role": "assistant", "content": answer},
    ]
    return answer, history


def _llm_chat(message, history=None):
    from openai import OpenAI

    if not LITELLM_API_KEY:
        raise RuntimeError("LITELLM_API_KEY is not set for CHAT_MODE=llm")

    history = history or []
    system = {
        "role": "system",
        "content": (
            "You are an analytics assistant for store CCTV footfall/demographics. "
            f"Current UTC time is {datetime.now(timezone.utc).isoformat()}. "
            "Only use tool results. Never invent numbers. Never identify individuals."
        ),
    }
    tools = [
        {
            "type": "function",
            "function": {
                "name": "list_stores",
                "description": "List stores",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_footfall",
                "description": "Unique footfall for a store",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "store_id": {"type": "string"},
                        "start": {"type": "string"},
                        "end": {"type": "string"},
                    },
                    "required": ["store_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_demographics",
                "description": "Gender/age breakdown for a store",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "store_id": {"type": "string"},
                        "start": {"type": "string"},
                        "end": {"type": "string"},
                    },
                    "required": ["store_id"],
                },
            },
        },
    ]

    def _window(start, end):
        end_dt = datetime.fromisoformat(end) if end else datetime.now(timezone.utc)
        start_dt = datetime.fromisoformat(start) if start else end_dt - timedelta(hours=24)
        return start_dt, end_dt

    dispatch = {
        "list_stores": lambda: fetch_stores(),
        "get_footfall": lambda store_id, start=None, end=None: (
            lambda s, e: {
                "store_id": store_id,
                "start": s.isoformat(),
                "end": e.isoformat(),
                "footfall": get_footfall_count(store_id, s, e),
            }
        )(*_window(start, end)),
        "get_demographics": lambda store_id, start=None, end=None: (
            lambda s, e: {
                "store_id": store_id,
                "start": s.isoformat(),
                "end": e.isoformat(),
                **get_demographics_breakdown(store_id, s, e),
            }
        )(*_window(start, end)),
    }

_client = None


def _get_client():
    global _client
    if _client is None:
        if not LITELLM_API_KEY:
            raise RuntimeError("LITELLM_API_KEY is not set - copy .env.example to .env and fill it in")
        _client = OpenAI(api_key=LITELLM_API_KEY, base_url=LITELLM_BASE_URL)
    return _client


def chat(message, history=None):
    """Run one user turn through the tool-calling loop. Returns (answer, new_history)."""
    history = history or []
    system = {"role": "system", "content": SYSTEM_PROMPT.format(now=datetime.now(timezone.utc).isoformat())}
    messages = [system] + history + [{"role": "user", "content": message}]

    for _ in range(MAX_TOOL_ROUNDS):
        response = client.chat.completions.create(
            model=LITELLM_MODEL, messages=messages, tools=tools
        )
        choice = response.choices[0].message
        if not choice.tool_calls:
            messages.append({"role": "assistant", "content": choice.content})
            return choice.content, messages[1:]

        messages.append({
            "role": "assistant",
            "content": choice.content,
            "tool_calls": [tc.model_dump() for tc in choice.tool_calls],
        })
        for tool_call in choice.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments or "{}")
            try:
                result = dispatch[name](**args)
            except Exception as e:
                result = {"error": str(e)}
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, default=str),
            })

    return "Could not finish that — try a narrower question.", messages[1:]


def chat(message, history=None):
    """Run one user turn. Returns (answer, new_history)."""
    if CHAT_MODE == "llm":
        try:
            return _llm_chat(message, history)
        except Exception as e:
            # Fall back so /chat still works when the proxy is down.
            answer, hist = _simple_chat(message, history)
            return (
                f"(LLM unavailable: {e}. Using simple mode.)\n{answer}",
                hist,
            )
    return _simple_chat(message, history)


if __name__ == "__main__":
    print(f"CCTV analytics chatbot (mode={CHAT_MODE})")
    print("Ask about footfall, demographics, or stores. Type 'exit' to quit.\n")
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
        answer, conversation = chat(user_input, conversation)
        print(f"bot> {answer}\n")
