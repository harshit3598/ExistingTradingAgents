"""Storing provider API keys the web form collects, without echoing them back.

Mirrors ``cli.prompts.ensure_api_key``'s persistence rules: the .env file holds
credentials, so it is created 0600 and tightened before every write.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import find_dotenv, set_key


def dotenv_path() -> Path:
    """Where keys are persisted (creating nothing)."""
    return Path(find_dotenv(usecwd=True) or Path.cwd() / ".env")


def save_api_key(env_var: str, key: str) -> str:
    """Persist ``key`` as ``env_var`` in .env (0600) and export it for this process."""
    if not env_var:
        raise ValueError("no environment variable known for that provider")
    key = str(key).strip()
    if not key:
        raise ValueError("the key is empty")
    if len(key) > 1024 or any(ch.isspace() or not ch.isprintable() for ch in key):
        raise ValueError("that does not look like an API key (no spaces or control characters)")

    env_path = dotenv_path()
    if not env_path.exists():
        env_path.parent.mkdir(parents=True, exist_ok=True)
        os.close(os.open(env_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
    os.chmod(env_path, 0o600)
    set_key(str(env_path), env_var, key)
    os.environ[env_var] = key
    return str(env_path)
