#!/usr/bin/env python3
"""Create .env from .env.example in one step.

Asks for the few values only a person can provide (domain, product name,
dashboard login, API keys), generates every secret, hashes the dashboard
password and writes .env (mode 600). Standard library only; works on Linux,
macOS and Windows.

    python3 setup_env.py            # interactive
    python3 setup_env.py --force    # overwrite an existing .env
"""

import argparse
import getpass
import hashlib
import os
import platform
import re
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXAMPLE = ROOT / ".env.example"
TARGET = ROOT / ".env"

# Generated, never typed: each gets its own random value.
SECRETS = (
    "POSTGRES_PASSWORD",
    "WEBUI_SECRET_KEY",
    "EXTERNAL_LOADER_API_KEY",
    "ADMIN_API_KEY",
    "DATA_API_KEY",
    "OPEN_TERMINAL_API_KEY",
    "PHOENIX_SECRET",
    "PHOENIX_DEFAULT_ADMIN_INITIAL_PASSWORD",
    "SESSION_SECRET",
)

# (variable, question, required)
API_KEYS = (
    ("EMBEDDING_API_KEY", "Voyage AI key for embeddings", True),
    ("GOOGLE_API_KEY", "Google AI Studio (Gemini) key", False),
    ("ANTHROPIC_API_KEY", "Anthropic (Claude) key", False),
    ("GROQ_API_KEY", "Groq key (audio/video transcription)", False),
    ("RERANKER_API_KEY", "Voyage AI key for reranking (often the same as embeddings)", False),
)

PBKDF2_ITERATIONS = 310000


def password_hash(password: str) -> str:
    """Same format the dashboard verifies: pbkdf2_sha256:<iterations>:<salt>:<hex>."""
    salt = os.urandom(16).hex()
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), PBKDF2_ITERATIONS).hex()
    return "pbkdf2_sha256:%d:%s:%s" % (PBKDF2_ITERATIONS, salt, digest)


def ask(question: str, default: str = "", required: bool = False) -> str:
    suffix = " [%s]" % default if default else (" (required)" if required else " (Enter to skip)")
    while True:
        value = input("%s%s: " % (question, suffix)).strip() or default
        if value or not required:
            return value
        print("  This value is required.")


def ask_password() -> str:
    while True:
        first = getpass.getpass("Dashboard admin password (min 10 characters): ")
        if len(first) < 10:
            print("  Too short.")
            continue
        if getpass.getpass("Repeat the password: ") == first:
            return first
        print("  The passwords do not match.")


def fill(template: str, values: dict) -> str:
    """Replace the value of every KEY= line named in `values`; keep everything else."""
    def repl(match):
        key = match.group(1)
        return "%s=%s" % (key, values[key]) if key in values else match.group(0)
    return re.sub(r"^([A-Z_][A-Z0-9_]*)=.*$", repl, template, flags=re.M)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="overwrite an existing .env")
    args = parser.parse_args()

    if not EXAMPLE.exists():
        print("error: %s not found - run this from the repository root" % EXAMPLE, file=sys.stderr)
        return 1
    if TARGET.exists() and not args.force:
        print("error: .env already exists. Use --force to overwrite it.", file=sys.stderr)
        return 1

    print("Creates .env for Growlyze Multimodal AI Chat. Secrets are generated for you.\n")
    domain = ask("Domain (public hostname, or localhost for a local run)", "localhost")
    app_name = ask("Product name shown to users", "Growlyze Multimodal AI Chat")
    stack = ask("Short stack name (letters/digits, unique per client on one machine)", "gmac")
    username = ask("Dashboard admin username", "admin")
    password = ask_password()

    on_linux = platform.system() == "Linux"
    values = {
        "DOMAIN": domain,
        "PUBLIC_BASE_URL": "https://%s" % domain,
        "APP_NAME": app_name,
        "STACK_NAME": stack,
        "ADMIN_USERNAME": username,
        "ADMIN_PASSWORD_HASH": password_hash(password),
        # Docker Desktop (macOS/Windows) can only share folders under the user's
        # directories, so keep data next to the repository there.
        "DATA_DIR": "/data" if on_linux else "./data",
        "RCLONE_CONFIG_DIR": "/root/.config/rclone" if on_linux else "./data/rclone",
    }
    values.update({key: secrets.token_hex(32) for key in SECRETS})

    print("\nAPI keys (section 7 of README.md explains where to get each one):")
    for key, question, required in API_KEYS:
        value = ask(question, required=required)
        if value:
            values[key] = value

    TARGET.write_text(fill(EXAMPLE.read_text(encoding="utf-8"), values), encoding="utf-8")
    try:
        TARGET.chmod(0o600)
    except OSError:
        pass

    print("\n.env written. Next steps:")
    print("  1. docker compose build && docker compose up -d")
    print("  2. Open https://%s/ and create the first (admin) account" % domain)
    print("  3. Finish the Open WebUI setup in README.md section 5. The two keys it asks for")
    print("     are in .env: EXTERNAL_LOADER_API_KEY and OPEN_TERMINAL_API_KEY")
    print("  4. Admin dashboard: https://%s/kb/  (user: %s)" % (domain, username))
    return 0


if __name__ == "__main__":
    sys.exit(main())
