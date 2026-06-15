"""Kalshi RSA-PSS request signing.

Kalshi's trading API authenticates each request with an RSA-PSS signature over
``timestamp_ms + HTTP_METHOD + path`` (path excludes the query string). If no
key is configured the scanner falls back to public, unauthenticated reads.

Environment:
    KALSHI_KEY_ID            API key id (UUID)
    KALSHI_PRIVATE_KEY_PATH  path to the RSA private key PEM file
"""

from __future__ import annotations

import base64
import os
import sys
import time


class KalshiAuth:
    def __init__(self) -> None:
        self.key_id: str = os.getenv("KALSHI_KEY_ID", "")
        self.key_path: str = os.getenv("KALSHI_PRIVATE_KEY_PATH", "")
        self._private_key = None
        if self.key_id and self.key_path:
            self._load_key()

    def _load_key(self) -> None:
        try:
            from cryptography.hazmat.primitives import serialization

            with open(self.key_path, "rb") as f:
                self._private_key = serialization.load_pem_private_key(
                    f.read(), password=None
                )
        except Exception as e:  # missing file / bad key -> fall back to public
            print(f"[KalshiAuth] could not load private key: {e}", file=sys.stderr)
            self._private_key = None

    @property
    def enabled(self) -> bool:
        return bool(self.key_id and self._private_key is not None)

    def _sign(self, message: str) -> str:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding

        signature = self._private_key.sign(
            message.encode("utf-8"),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode("utf-8")

    def headers(self, method: str, path: str) -> dict[str, str]:
        """Signed headers for a request. ``path`` must exclude any query string."""
        if not self.enabled:
            return {}
        timestamp_ms = str(int(time.time() * 1000))
        message = timestamp_ms + method.upper() + path
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
            "KALSHI-ACCESS-SIGNATURE": self._sign(message),
        }
