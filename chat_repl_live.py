"""Interactive REPL for chatbot.py against the REAL Postgres data and the
REAL LiteLLM proxy (CHAT_MODE/LITELLM_*/DB_* all come from .env in this
directory - forced with override=True so stray session env vars from
earlier local docker-compose testing, e.g. DB_PORT=5433, don't win). Must
be run somewhere with network access to litellm.internal.givadiva.co (e.g.
on the office VPN) - it will time out and fall back to simple mode
otherwise.

Run: <path-to-python> chat_repl_live.py
Type 'exit' or 'quit' to stop.
"""
import os
import sys

_HERE = __file__.rsplit("\\", 1)[0]
sys.path.insert(0, _HERE)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(_HERE, ".env"), override=True)

import chatbot  # noqa: E402

print(f"CCTV analytics chatbot (mode={chatbot.CHAT_MODE}) - live Postgres + LiteLLM")
print("Try: 'which stores do we have', 'footfall for store20 today', "
      "'gender breakdown for store981'")
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
