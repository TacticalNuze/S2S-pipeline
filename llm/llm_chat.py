"""
The LLM part of the pipeline: any OpenAI-compatible endpoint, configured in .env
(this folder or any parent):

    OPENAI_API_KEY=...
    OPENAI_BASE_URL=...      # optional, for non-OpenAI providers (Ollama, LM Studio, ...)
    OPENAI_MODEL=gpt-4.1     # optional

SYSTEM_PROMPT is shared by full-run/bot_habibi.py and full-run/habibi_adapter.py.

Text chat with the same prompt, to check the LLM settings without any audio:

    uv run llm/llm_chat.py                      # interactive, empty line to quit
    uv run llm/llm_chat.py "السلام، شنو سميتك؟"   # one question
"""
import argparse
import os
import sys

import truststore
from dotenv import load_dotenv

# Verify HTTPS with the Windows certificate store (corporate proxy breaks requests otherwise).
truststore.inject_into_ssl()
load_dotenv()

SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Reply in Moroccan Darija written in Arabic script, "
    "in one or two short sentences. Your reply is spoken aloud, so use no emojis, lists, "
    "Latin letters or other formatting."
)


def model() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-4.1")


def client():
    from openai import OpenAI
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL") or None)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Arabic on a Windows console
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", nargs="?", help="ask one question and exit")
    args = parser.parse_args()

    llm = client()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    print(f"model: {model()} at {llm.base_url}")
    while True:
        question = args.question or input("\nyou: ").strip()
        if not question:
            return
        messages.append({"role": "user", "content": question})
        print("bot: ", end="", flush=True)
        reply = ""
        for event in llm.chat.completions.create(model=model(), messages=messages, stream=True):
            if event.choices:
                delta = event.choices[0].delta.content or ""
                reply += delta
                print(delta, end="", flush=True)
        print()
        messages.append({"role": "assistant", "content": reply})
        if args.question:
            return


if __name__ == "__main__":
    main()
