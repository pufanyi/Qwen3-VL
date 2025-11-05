from __future__ import annotations

import re
from typing import Callable, List, Optional, Pattern, Sequence, Tuple

from loguru import logger

try:
    from aoss_client.client import Client as AOSSClient  # type: ignore
except ImportError:
    AOSSClient = None  # type: ignore[misc,assignment]


class PatternAOSSClient:
    """Proxy AOSS client that routes requests based on URI patterns."""

    _DEFAULT_KEY = "__DEFAULT__"

    def __init__(
        self,
        default_conf_path: Optional[str],
        pattern_rules: Optional[Sequence[Tuple[str, str]]],
        client_factory: Optional[Callable[[Optional[str]], object]] = None,
    ) -> None:
        if AOSSClient is None and client_factory is None:
            raise ImportError("AOSS SDK not installed. Please install it or provide a client_factory.")

        self._client_factory = client_factory or self._build_client
        self._default_conf_path = default_conf_path
        self._clients: dict[str, object] = {}

        self._compiled_rules: List[Tuple[Pattern[str], str]] = []
        for pattern, conf_path in pattern_rules or []:
            try:
                compiled = re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"Invalid regex pattern for AOSS rule '{pattern}': {exc}") from exc
            self._compiled_rules.append((compiled, conf_path))

        if self._compiled_rules:
            logger.info(
                "Initialized PatternAOSSClient with %d pattern rule(s); default config: %s",
                len(self._compiled_rules),
                default_conf_path or "<default>",
            )

    @staticmethod
    def _build_client(conf_path: Optional[str]) -> object:
        if conf_path:
            return AOSSClient(conf_path)  # type: ignore[call-arg]
        return AOSSClient()  # type: ignore[call-arg]

    def _get_or_create_client(self, conf_path: Optional[str]) -> object:
        key = conf_path or self._DEFAULT_KEY
        if key not in self._clients:
            self._clients[key] = self._client_factory(conf_path)
        return self._clients[key]

    def _select_conf(self, key: str) -> Optional[str]:
        for pattern, conf_path in self._compiled_rules:
            if pattern.search(key):
                return conf_path
        return self._default_conf_path

    def get(self, key: str, *args, **kwargs):
        conf_path = self._select_conf(key)
        client = self._get_or_create_client(conf_path)
        # logger.debug(f"AOSS request routed to config '{conf_path or '<default>'}' for key '{key}'")
        return getattr(client, "get")(key, *args, **kwargs)

    def __getattr__(self, item: str):
        default_client = self._get_or_create_client(self._default_conf_path)
        return getattr(default_client, item)
