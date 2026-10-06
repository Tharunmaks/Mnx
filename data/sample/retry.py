import time


def retry(func, attempts=3, delay=0.1):
    """Call func until it succeeds or attempts are exhausted."""
    last_error = None
    for _ in range(attempts):
        try:
            return func()
        except Exception as exc:
            last_error = exc
            time.sleep(delay)
    raise last_error
