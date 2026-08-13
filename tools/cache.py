"""
src/tools/cache.py
--------------------
Every Tavily search and ScrapeGraph scrape costs money and time. During
development you will re-run the same crew a dozen times against the same
product category -- without caching, that's a dozen redundant API calls.

This is a minimal disk-backed cache, keyed by a hash of (function name +
arguments), with a TTL. It is intentionally NOT a general-purpose caching
framework: it's ~40 lines because that's all this problem needs, and a
simpler cache is easier to reason about when debugging "why is this crew
returning stale data".
"""
from __future__ import annotations

import functools
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

from src.config import CACHE_DIR, settings


def _cache_key(prefix: str, *args: Any, **kwargs: Any) -> Path:
    raw = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    return CACHE_DIR / f"{prefix}_{digest}.json"


def cached(prefix: str):
    """
    Decorator for plain functions that return JSON-serializable data.
    Usage:
        @cached("tavily_search")
        def run_search(query: str) -> dict: ...
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            path = _cache_key(prefix, *args, **kwargs)
            ttl_seconds = settings.request_cache_ttl_hours * 3600

            if path.exists():
                age = time.time() - path.stat().st_mtime
                if age < ttl_seconds:
                    try:
                        return json.loads(path.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        pass  # corrupt cache entry -> fall through and refetch

            result = fn(*args, **kwargs)
            try:
                path.write_text(json.dumps(result, default=str), encoding="utf-8")
            except (TypeError, OSError):
                pass  # non-serializable or disk issue -> caching is best-effort
            return result

        return wrapper
    return decorator


def clear_cache() -> int:
    """Utility for `main.py --clear-cache`. Returns number of files removed."""
    count = 0
    for f in CACHE_DIR.glob("*.json"):
        f.unlink(missing_ok=True)
        count += 1
    return count
