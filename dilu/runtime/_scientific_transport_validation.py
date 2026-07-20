from __future__ import annotations

import math
import re
from urllib.parse import urlparse


FULL_MODEL_DIGEST_PATTERN = re.compile(r"\Asha256:[0-9a-f]{64}\Z")
UINT32_MAX = (1 << 32) - 1


def require_native_endpoint(value: str) -> None:
    require_canonical_text("native_endpoint", value)
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("native_endpoint must be an absolute HTTP(S) URL.")
    if parsed.path.rstrip("/") != "/api/chat" or parsed.query or parsed.fragment:
        raise ValueError("Scientific transport requires the native /api/chat endpoint.")


def require_canonical_text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical string.")


def require_model_digest(name: str, value: str) -> None:
    if not isinstance(value, str) or not FULL_MODEL_DIGEST_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must be a canonical full sha256 digest.")


def require_bool(name: str, value: bool) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean.")


def require_nonnegative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")


def require_positive_int(name: str, value: int) -> None:
    require_nonnegative_int(name, value)
    if value == 0:
        raise ValueError(f"{name} must be positive.")


def require_uint32(name: str, value: int) -> None:
    require_nonnegative_int(name, value)
    if value > UINT32_MAX:
        raise ValueError(f"{name} must fit in uint32.")


def require_optional_nonnegative_int(name: str, value: int | None) -> None:
    if value is not None:
        require_nonnegative_int(name, value)


def require_nonnegative_number(name: str, value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0
    ):
        raise ValueError(f"{name} must be non-negative numeric.")
