"""Polite HTTP client: rate limit, retries with exponential backoff + jitter, on-disk cache."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from joblens.config import USER_AGENT

log = logging.getLogger(__name__)

RETRY_STATUS = {429, 500, 502, 503, 504}


class RetryableHTTPError(Exception):
    """Raised for status codes worth retrying (429 / 5xx)."""


def _is_retryable(exc: BaseException) -> bool:
    return isinstance(exc, RetryableHTTPError | httpx.TransportError)


def cache_key(url: str, params: Mapping[str, Any] | None) -> str:
    payload = json.dumps({"url": url, "params": dict(params or {})}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


class PoliteClient:
    """GET-only JSON client. One request per `min_interval` seconds, cached on disk.

    The cache directory holds raw API responses and therefore must be git-ignored
    (default lives under data/raw/).
    """

    def __init__(
        self,
        cache_dir: Path,
        min_interval: float = 1.0,
        headers: Mapping[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
        max_attempts: int = 5,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval = min_interval
        self._sleep = sleep
        self._clock = clock
        self._last: float | None = None
        self.network_calls = 0
        self.cache_hits = 0
        self._client = httpx.Client(
            headers={"User-Agent": USER_AGENT, "Accept": "application/json", **(headers or {})},
            timeout=httpx.Timeout(30.0),
            follow_redirects=True,
            transport=transport,
        )
        self._get_with_retry = retry(
            retry=retry_if_exception(_is_retryable),
            wait=wait_random_exponential(multiplier=1, max=60),
            stop=stop_after_attempt(max_attempts),
            reraise=True,
            sleep=sleep,
        )(self._get_once)

    def _throttle(self) -> None:
        now = self._clock()
        if self._last is not None:
            wait = self.min_interval - (now - self._last)
            if wait > 0:
                self._sleep(wait)
        self._last = self._clock()

    def _get_once(self, url: str, params: Mapping[str, Any] | None) -> Any:
        self._throttle()
        self.network_calls += 1
        resp = self._client.get(url, params=dict(params or {}))
        if resp.status_code in RETRY_STATUS:
            raise RetryableHTTPError(f"{resp.status_code} for {url}")
        resp.raise_for_status()
        return resp.json()

    def get_json(self, url: str, params: Mapping[str, Any] | None = None) -> Any:
        path = self.cache_dir / f"{cache_key(url, params)}.json"
        if path.exists():
            self.cache_hits += 1
            return json.loads(path.read_text(encoding="utf-8"))
        data = self._get_with_retry(url, params)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
        return data

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> PoliteClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
