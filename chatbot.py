"""
Natural-language chat over CCTV analytics data.

Default mode (CHAT_MODE=simple) is a small keyword/intent router that calls
the same Postgres helpers as api.py (footfall, demographics, dwell-time,
persons, entry/exit, stores) - no external LLM required.

Optional mode (CHAT_MODE=llm) runs a tool-calling loop against a LiteLLM
proxy, using the same helper functions as tools rather than raw SQL. If the
proxy is unreachable, /chat falls back to simple mode rather than failing.
"""
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

from db.postgres import (
    fetch_stores,
    get_average_dwell_time,
    get_demographics_breakdown,
    get_footfall_count,
    list_entry_exit_logs,
    list_persons,
)

load_dotenv()

# CLIP + the visual-search Chroma collection live in analytics_service/src
# (same code that ingests them from the pipeline) - reach them the same way
# persist.py reaches `db` from the other side.
sys.path.insert(0, str(Path(__file__).resolve().parent / "analytics_service" / "src"))

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
_DWELL_RE = re.compile(r"\b(dwell|time spent|how long|visit duration|time in store)\b", re.I)
_VISUAL_TRIGGER_RE = re.compile(r"\b(find|show|search for|look for|looking for)\b", re.I)
_VISUAL_DESC_RE = re.compile(
    r"\b(wearing|dressed|carrying|holding|colou?r|shirt|jacket|clothing|outfit|hat|bag)\b", re.I
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


def _fmt_duration(seconds):
    if seconds is None:
        return "n/a"
    minutes, secs = divmod(round(seconds), 60)
    return f"{minutes}m {secs}s" if minutes else f"{secs}s"


def _fmt_dwell(store_id, start, end, stats):
    if not stats or not stats.get("sample_size"):
        return f"No completed visits for {store_id} between {start.isoformat()} and {end.isoformat()}."
    return (
        f"Average time spent at {store_id} from {start.isoformat()} to {end.isoformat()}: "
        f"{_fmt_duration(stats['avg_seconds'])} "
        f"(median {_fmt_duration(stats['median_seconds'])}, "
        f"min {_fmt_duration(stats['min_seconds'])}, max {_fmt_duration(stats['max_seconds'])}, "
        f"n={stats['sample_size']} visits)."
    )


def _simple_chat(message: str, history=None):
    history = history or []
    text = (message or "").strip()
    if not text:
        return "Ask about stores, footfall, demographics, dwell time, persons, or entry/exit logs.", history

    if _IDENTIFY_RE.search(text):
        answer = (
            "I only answer aggregate analytics (footfall, demographics, visit counts, dwell time). "
            "Identifying or naming individuals is out of scope."
        )
        history = history + [
            {"role": "user", "content": text},
            {"role": "assistant", "content": answer},
        ]
        return answer, history

    if _VISUAL_TRIGGER_RE.search(text) and _VISUAL_DESC_RE.search(text):
        store_id = _resolve_store_id(text)
        try:
            matches = run_visual_search(text, store_id=store_id, top_k=5)
        except Exception as e:
            answer = f"Visual search unavailable: {e}"
        else:
            if not matches:
                answer = "No matching visits found (only visits captured with visual search enabled are searchable)."
            else:
                lines = [
                    f"- {m['similarity']:.2f} similarity, store={m['store_id']}, "
                    f"camera={m['camera_id']}, seen at {datetime.fromtimestamp(m['seen_at'], tz=timezone.utc).isoformat()}"
                    for m in matches
                ]
                answer = (
                    "Possible matches (ranked by appearance similarity, not exact - "
                    "thumbnails available via POST /visual-search):\n" + "\n".join(lines)
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
            "footfall", "visitor", "visitors", "unique", "demographic", "gender",
            "female", "male", "women", "men", "age", "person", "entry", "exit",
            "log", "came", "how many", "dwell", "time spent", "how long",
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

    if _DWELL_RE.search(text):
        stats = get_average_dwell_time(store_id, start, end)
        answer = _fmt_dwell(store_id, start, end, stats)
    elif gender_focus or any(
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
            "footfall", "unique visitor", "how many visitor", "visitor count",
            "how many people", "how many came", "people came",
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
            "I can answer: stores, footfall, demographics (gender/age), dwell time, "
            "persons, or entry/exit logs. Example: \"footfall for store2 today\" or "
            "\"average time spent at store2 this week\"."
        )

    history = history + [
        {"role": "user", "content": text},
        {"role": "assistant", "content": answer},
    ]
    return answer, history


SYSTEM_PROMPT = """You are an analytics assistant for a store CCTV footfall/demographics system.

Data model:
- Each "person" row is one re-identified VISIT, not a named individual - gender
  and age_group are model estimates with a confidence score, never verified
  identity. Never refer to a person by name or imply identification.
- "footfall" = count of distinct persons with an 'entry' event in a time window.
- "dwell time" / "time spent" = last_seen - first_seen per visit, store-level
  only (no per-zone breakdown yet).
- search_visual matches by CLIP appearance similarity, not exact attributes -
  present results as "possible matches, ranked by similarity", not certainties,
  and note that only visits captured with visual search enabled are covered.
- All timestamps are UTC. The current UTC time is {now}.
- Only answer using the tool results you get back - never invent numbers.
- If a question needs a store_id you don't already have, call list_stores first.
- If asked to identify, name, or track a specific individual beyond aggregate
  stats, decline - that's outside this system's scope by design.

Answer conversationally and cite the actual numbers the tools returned.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_visual",
            "description": (
                "Semantic appearance search: find visits matching a free-text visual "
                "description (e.g. 'a person wearing a red jacket'), via CLIP embeddings "
                "over recent visit thumbnails (only visits captured with visual search "
                "enabled, within VISUAL_SEARCH_TTL_DAYS, are searchable). Off-the-shelf "
                "CLIP on low-res CCTV crops - treat similarity as a coarse ranking signal, "
                "not a precise attribute match."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Visual description to search for"},
                    "store_id": {"type": "string"},
                    "start": {"type": "string"},
                    "end": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_stores",
            "description": "List all stores/cameras configured in the system (store_id, name, region).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_footfall",
            "description": "Unique footfall (distinct visitors) for a store in a time window.",
            "parameters": {
                "type": "object",
                "properties": {
                    "store_id": {"type": "string"},
                    "start": {"type": "string", "description": "ISO8601 UTC datetime; defaults to 24h before end"},
                    "end": {"type": "string", "description": "ISO8601 UTC datetime; defaults to now"},
                },
                "required": ["store_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_demographics",
            "description": "Gender and age-group breakdown of visitors for a store in a time window.",
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
            "name": "get_dwell_time",
            "description": "Average/median/min/max time spent per visit (store-level) in a time window.",
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
            "name": "list_persons",
            "description": "List individual visit records (person rows) for a store in a time window.",
            "parameters": {
                "type": "object",
                "properties": {
                    "store_id": {"type": "string"},
                    "start": {"type": "string"},
                    "end": {"type": "string"},
                    "limit": {"type": "integer", "default": 50},
                },
                "required": ["store_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_entry_exit_logs",
            "description": "List raw entry/exit/zone events for a store (optionally filtered to one person_id).",
            "parameters": {
                "type": "object",
                "properties": {
                    "store_id": {"type": "string"},
                    "person_id": {"type": "string"},
                    "start": {"type": "string"},
                    "end": {"type": "string"},
                    "limit": {"type": "integer", "default": 50},
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


def _tool_get_footfall(store_id, start=None, end=None):
    s, e = _window(start, end)
    return {"store_id": store_id, "start": s.isoformat(), "end": e.isoformat(),
            "footfall": get_footfall_count(store_id, s, e)}


def _tool_get_demographics(store_id, start=None, end=None):
    s, e = _window(start, end)
    return {"store_id": store_id, "start": s.isoformat(), "end": e.isoformat(),
            **get_demographics_breakdown(store_id, s, e)}


def _tool_get_dwell_time(store_id, start=None, end=None):
    s, e = _window(start, end)
    return {"store_id": store_id, "start": s.isoformat(), "end": e.isoformat(),
            **get_average_dwell_time(store_id, s, e)}


def _tool_list_persons(store_id, start=None, end=None, limit=50):
    s, e = _window(start, end)
    return list_persons(store_id, s, e, limit)


def _tool_list_entry_exit_logs(store_id, person_id=None, start=None, end=None, limit=50):
    s, e = _window(start, end)
    return list_entry_exit_logs(store_id, person_id, s, e, limit)


_visual_client = None


def run_visual_search(query, store_id=None, start=None, end=None, top_k=5):
    """Full results, including thumbnails - used by the /visual-search endpoint
    directly and (with thumbnails stripped) by the chatbot tool below.
    Loads CLIP + connects to Chroma lazily, only when a visual search actually runs."""
    global _visual_client
    import visual_embeddings
    from services.visual_search import VisualSearchClient

    if _visual_client is None:
        _visual_client = VisualSearchClient()

    s, e = _window(start, end) if (start or end) else (None, None)
    start_ts = int(s.timestamp()) if s else None
    end_ts = int(e.timestamp()) if e else None

    query_embedding = visual_embeddings.encode_text(query)
    return _visual_client.search_by_embedding(query_embedding, store_id, start_ts, end_ts, top_k)


def _tool_search_visual(query, store_id=None, start=None, end=None, top_k=5):
    results = run_visual_search(query, store_id, start, end, top_k)
    # Thumbnails are base64 image blobs - useless (and token-expensive) for the
    # LLM to read; the /visual-search endpoint returns them for display instead.
    return [{k: v for k, v in r.items() if k != "thumbnail_b64"} for r in results]


DISPATCH = {
    "list_stores": lambda: fetch_stores(),
    "get_footfall": _tool_get_footfall,
    "get_demographics": _tool_get_demographics,
    "get_dwell_time": _tool_get_dwell_time,
    "list_persons": _tool_list_persons,
    "list_entry_exit_logs": _tool_list_entry_exit_logs,
    "search_visual": _tool_search_visual,
}

_client = None


def _get_client():
    global _client
    if _client is None:
        if not LITELLM_API_KEY:
            raise RuntimeError("LITELLM_API_KEY is not set - copy .env.example to .env and fill it in")
        from openai import OpenAI

        _client = OpenAI(api_key=LITELLM_API_KEY, base_url=LITELLM_BASE_URL)
    return _client


def _llm_chat(message, history=None):
    history = history or []
    system = {"role": "system", "content": SYSTEM_PROMPT.format(now=datetime.now(timezone.utc).isoformat())}
    messages = [system] + history + [{"role": "user", "content": message}]
    client = _get_client()

    for _ in range(MAX_TOOL_ROUNDS):
        response = client.chat.completions.create(model=LITELLM_MODEL, messages=messages, tools=TOOLS)
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
                result = DISPATCH[name](**args)
            except Exception as e:
                result = {"error": str(e)}
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, default=str),
            })

    return "I couldn't finish that after several tool calls - try a narrower question.", messages[1:]


def chat(message, history=None):
    """Run one user turn. Returns (answer, new_history)."""
    if CHAT_MODE == "llm":
        try:
            return _llm_chat(message, history)
        except Exception as e:
            # Fall back so /chat still works when the proxy is unreachable.
            answer, hist = _simple_chat(message, history)
            return f"(LLM unavailable: {e}. Using simple mode.)\n{answer}", hist
    return _simple_chat(message, history)


if __name__ == "__main__":
    print(f"CCTV analytics chatbot (mode={CHAT_MODE})")
    print("Ask about stores, footfall, demographics, dwell time, or entry/exit logs. Type 'exit' to quit.\n")
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
