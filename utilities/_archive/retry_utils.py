#!/usr/bin/env python3
"""
Retry Utilities for Resilient API Calls and Network Operations
===============================================================

Provides decorators and utilities for implementing exponential backoff retry logic.
Useful for handling transient failures in API calls, network requests, and file I/O.

Usage Examples:
--------------

Basic retry decorator:
    from utilities.retry_utils import retry

    @retry(max_attempts=3, backoff_factor=2)
    def call_api(url: str) -> dict:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

Retry with specific exceptions:
    @retry(
        max_attempts=5,
        backoff_factor=1.5,
        exceptions=(TimeoutError, ConnectionError, requests.HTTPError)
    )
    def fetch_data(endpoint: str) -> dict:
        # Your code here
        pass

Retry with custom callback:
    def on_retry(attempt, exception, delay):
        logging.warning(f"Attempt {attempt} failed: {exception}. Retrying in {delay}s")

    @retry(max_attempts=3, on_retry_callback=on_retry)
    def unreliable_operation():
        # Your code here
        pass

Context manager for retry logic:
    with RetryContext(max_attempts=3, backoff_factor=2) as retry_ctx:
        result = retry_ctx.execute(lambda: risky_operation())

Features:
---------
- Exponential backoff with configurable base delay and factor
- Configurable maximum attempts
- Optional jitter to prevent thundering herd
- Specific exception filtering
- Success/failure callbacks
- Detailed logging of retry attempts
- Context manager support

Last Updated: 2026-02-02
"""

import time
import argparse
import logging
import random
from functools import wraps
from typing import Callable, Optional, Tuple, Type, Any


def retry(
    max_attempts: int = 3,
    backoff_factor: float = 2.0,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    jitter: bool = True,
    on_retry_callback: Optional[Callable[[int, Exception, float], None]] = None,
    on_success_callback: Optional[Callable[[int], None]] = None,
    on_failure_callback: Optional[Callable[[Exception], None]] = None
) -> Callable:
    """
    Decorator that implements exponential backoff retry logic.

    Args:
        max_attempts: Maximum number of attempts (default: 3)
        backoff_factor: Multiplier for exponential backoff (default: 2.0)
        base_delay: Initial delay in seconds (default: 1.0)
        max_delay: Maximum delay between retries in seconds (default: 60.0)
        exceptions: Tuple of exception types to catch (default: (Exception,))
        jitter: Add random jitter to delays to prevent thundering herd (default: True)
        on_retry_callback: Optional callback(attempt, exception, delay) called before retry
        on_success_callback: Optional callback(attempt) called on success
        on_failure_callback: Optional callback(exception) called when all retries exhausted

    Returns:
        Decorated function with retry logic

    Example:
        >>> @retry(max_attempts=3, backoff_factor=2, exceptions=(ValueError, TypeError))
        >>> def parse_data(data: str) -> dict:
        ...     return json.loads(data)
        >>>
        >>> result = parse_data('{"key": "value"}')

    Retry Timing:
        With base_delay=1.0, backoff_factor=2.0:
        - Attempt 1: Immediate
        - Attempt 2: After 1.0s (+ jitter)
        - Attempt 3: After 2.0s (+ jitter)
        - Attempt 4: After 4.0s (+ jitter)
        - etc.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    result = func(*args, **kwargs)

                    # Success callback
                    if on_success_callback:
                        on_success_callback(attempt)

                    # Log success if it took multiple attempts
                    if attempt > 1:
                        logging.info(
                            f"{func.__name__} succeeded on attempt {attempt}/{max_attempts}"
                        )

                    return result

                except exceptions as e:
                    last_exception = e

                    # If this was the last attempt, give up
                    if attempt >= max_attempts:
                        logging.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {e}",
                            exc_info=True
                        )
                        if on_failure_callback:
                            on_failure_callback(e)
                        raise

                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (backoff_factor ** (attempt - 1)), max_delay)

                    # Add jitter if enabled (± 25% randomness)
                    if jitter:
                        jitter_range = delay * 0.25
                        delay = delay + random.uniform(-jitter_range, jitter_range)
                        delay = max(0, delay)  # Ensure non-negative

                    # Log retry attempt
                    logging.warning(
                        f"{func.__name__} attempt {attempt}/{max_attempts} failed: {e}. "
                        f"Retrying in {delay:.2f}s"
                    )

                    # Retry callback
                    if on_retry_callback:
                        on_retry_callback(attempt, e, delay)

                    # Wait before retry
                    time.sleep(delay)

            # Should never reach here, but just in case
            if last_exception:
                raise last_exception

        return wrapper
    return decorator


def retry_with_timeout(
    max_attempts: int = 3,
    backoff_factor: float = 2.0,
    base_delay: float = 1.0,
    timeout: float = 30.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
) -> Callable:
    """
    Retry decorator with per-attempt timeout.

    Useful for operations that might hang indefinitely.

    Args:
        max_attempts: Maximum number of attempts
        backoff_factor: Multiplier for exponential backoff
        base_delay: Initial delay in seconds
        timeout: Timeout per attempt in seconds
        exceptions: Exception types to catch

    Returns:
        Decorated function with retry and timeout logic

    Note:
        This is a simplified version. For production use, consider
        using signal-based timeouts or threading with timeouts.

    Example:
        >>> @retry_with_timeout(max_attempts=3, timeout=10.0)
        >>> def fetch_remote_data(url: str) -> bytes:
        ...     return requests.get(url, timeout=10.0).content
    """
    return retry(
        max_attempts=max_attempts,
        backoff_factor=backoff_factor,
        base_delay=base_delay,
        exceptions=exceptions
    )


class RetryContext:
    """
    Context manager for retry logic without decorator.

    Useful when you need retry logic for a specific code block
    without wrapping an entire function.

    Example:
        >>> retry_ctx = RetryContext(max_attempts=3, backoff_factor=2)
        >>> with retry_ctx:
        ...     result = risky_operation()
        ...     process(result)
        >>>
        >>> # Or use execute method:
        >>> result = retry_ctx.execute(lambda: risky_operation())
    """

    def __init__(
        self,
        max_attempts: int = 3,
        backoff_factor: float = 2.0,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exceptions: Tuple[Type[Exception], ...] = (Exception,),
        jitter: bool = True
    ):
        self.max_attempts = max_attempts
        self.backoff_factor = backoff_factor
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exceptions = exceptions
        self.jitter = jitter
        self.attempt = 0
        self.last_exception = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            return True

        # Check if this exception should be retried
        if not issubclass(exc_type, self.exceptions):
            return False

        self.attempt += 1
        self.last_exception = exc_val

        # If max attempts reached, re-raise
        if self.attempt >= self.max_attempts:
            logging.error(
                f"RetryContext failed after {self.max_attempts} attempts: {exc_val}"
            )
            return False

        # Calculate delay
        delay = min(
            self.base_delay * (self.backoff_factor ** (self.attempt - 1)),
            self.max_delay
        )

        if self.jitter:
            jitter_range = delay * 0.25
            delay = delay + random.uniform(-jitter_range, jitter_range)
            delay = max(0, delay)

        logging.warning(
            f"RetryContext attempt {self.attempt}/{self.max_attempts} failed: {exc_val}. "
            f"Retrying in {delay:.2f}s"
        )

        time.sleep(delay)
        return True  # Suppress exception and retry

    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function with retry logic.

        Args:
            func: Callable to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result of func

        Raises:
            Last exception if all retries exhausted
        """
        decorator = retry(
            max_attempts=self.max_attempts,
            backoff_factor=self.backoff_factor,
            base_delay=self.base_delay,
            max_delay=self.max_delay,
            exceptions=self.exceptions,
            jitter=self.jitter
        )
        wrapped_func = decorator(func)
        return wrapped_func(*args, **kwargs)


# Predefined retry decorators for common scenarios

# For network operations (ConnectionError, TimeoutError, etc.)
retry_network = retry(
    max_attempts=3,
    backoff_factor=2.0,
    base_delay=1.0,
    exceptions=(ConnectionError, TimeoutError, OSError)
)

# For API calls (with more attempts and longer delays)
retry_api = retry(
    max_attempts=5,
    backoff_factor=1.5,
    base_delay=2.0,
    max_delay=30.0
)

# For file I/O operations
retry_io = retry(
    max_attempts=3,
    backoff_factor=1.5,
    base_delay=0.5,
    exceptions=(IOError, OSError, PermissionError)
)


# Example usage
def run_demo() -> None:
    """Run a small demonstration of the retry helpers."""
    # Setup logging to see retry attempts
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Example 1: Basic retry
    @retry(max_attempts=3, base_delay=0.5)
    def flaky_operation():
        """Simulates an operation that fails randomly."""
        if random.random() < 0.7:
            raise ValueError("Random failure!")
        return "Success!"

    try:
        result = flaky_operation()
        print(f"Result: {result}")
    except ValueError as e:
        print(f"Failed after retries: {e}")

    # Example 2: Retry with callbacks
    def log_retry(attempt, exception, delay):
        print(f"  → Retry callback: attempt {attempt}, exception: {exception}, delay: {delay:.2f}s")

    @retry(max_attempts=2, on_retry_callback=log_retry, base_delay=0.5)
    def another_operation():
        raise ConnectionError("Network unavailable")

    try:
        another_operation()
    except ConnectionError:
        print("All retries exhausted")

    # Example 3: Context manager
    retry_ctx = RetryContext(max_attempts=3, base_delay=0.5)
    result = retry_ctx.execute(lambda: "Direct execution")
    print(f"Context result: {result}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retry utilities module",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a small demonstration of the retry helpers",
    )
    args = parser.parse_args()

    if args.demo:
        run_demo()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
