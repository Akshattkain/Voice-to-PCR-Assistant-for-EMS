"""Latency measurement utilities."""

import time
from functools import wraps
from typing import Any, Callable

from app.utils.logging import logger


def measure_latency(func: Callable) -> Callable:
    """Decorator to measure and log function execution time in milliseconds."""

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        # Handle both sync and async
        if hasattr(result, "__await__"):
            result = await result
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(f"{func.__qualname__} completed in {elapsed_ms:.1f}ms")
        return result

    return wrapper
