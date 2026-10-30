"""Voyage key check: 2 endpoints x 2 models, for one or more keys.

    python check_voyage_key.py                    # key from .env
    python check_voyage_key.py KEY1 KEY2          # compare keys side by side

Keys come from argv, else $VOYAGE_API_KEY, else EMBEDDING_API_KEY in ./.env.
"""

import os
import sys

import httpx
from dotenv import load_dotenv

ENDPOINTS = {
    "MongoDB Atlas": "https://ai.mongodb.com/v1/embeddings",
    "Voyage direct": "https://api.voyageai.com/v1/embeddings",
}
MODELS = ["voyage-code-4", "voyage-3"]

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
KEYS = sys.argv[1:] or [k for k in (os.getenv("VOYAGE_API_KEY"),
                                    os.getenv("EMBEDDING_API_KEY")) if k]


def probe(url, model, key):
    try:
        r = httpx.post(url, json={"input": ["test"], "model": model},
                       headers={"Content-Type": "application/json",
                                "Authorization": f"Bearer {key}"}, timeout=30.0)
    except httpx.HTTPError as exc:
        print(f"  {model:<15} unreachable: {exc}")
        return False
    body = r.text[:100].replace("\n", " ")
    ok = r.status_code == 200
    print(f"  {model:<15} {r.status_code}  {'OK  ' if ok else 'FAIL'}  {body}")
    return ok


def check(key):
    print(f"KEY {key[:6]}...{key[-4:]}  ({len(key)} chars)\n")
    working = []
    for name, url in ENDPOINTS.items():
        print(f"{name}  {url}")
        for model in MODELS:
            if probe(url, model, key):
                working.append((url, model))
        print()
    if working:
        url, model = working[0]
        print(f"Use this endpoint: {url}")
        print(f"Use this model name: {model}")
        if len(working) > 1:
            print(f"(also worked: {', '.join(f'{m} @ {u}' for u, m in working[1:])})")
        return True
    print("Nothing worked - key rejected by both hosts.")
    print("401 = key invalid/revoked. 403 = key belongs to the other platform.")
    return False


def main():
    if not KEYS:
        print("No key. Pass one or more as arguments, set VOYAGE_API_KEY, "
              "or put EMBEDDING_API_KEY in .env")
        return 1
    results = []
    for i, key in enumerate(KEYS):
        if i:
            print("\n" + "=" * 62 + "\n")
        results.append((key, check(key)))
    if len(results) > 1:
        print("\n" + "=" * 62 + "\nSUMMARY")
        for key, ok in results:
            print(f"  {key[:6]}...{key[-4:]}  {'WORKS' if ok else 'REJECTED'}")
    return 0 if any(ok for _, ok in results) else 1


if __name__ == "__main__":
    sys.exit(main())
