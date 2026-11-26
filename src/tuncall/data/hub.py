"""Hub downloads that survive network drops (D-16g).

A failed download is retried after 30 s, doubling up to 10 min between tries, for up to
12 hours. Finished files stay in the Hub cache and partial files resume, so a retry only
fetches what is missing. Answers that a retry cannot change (4xx: wrong repo, revision,
file or token) are raised at once.
"""

import logging
import time
from collections.abc import Callable

from huggingface_hub.errors import HfHubHTTPError

log = logging.getLogger(__name__)

FIRST_WAIT = 30.0
MAX_WAIT = 600.0
GIVE_UP_AFTER = 12 * 3600.0


def permanent(err: Exception) -> bool:
    """A 4xx answer from the Hub, except timeout (408) and rate limit (429)."""
    if not isinstance(err, HfHubHTTPError):
        return False
    status = err.response.status_code
    return 400 <= status < 500 and status not in (408, 429)


def retrying[T](
    fn: Callable[[], T],
    *,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> T:
    """``fn()``, retried on network errors (see the module docstring)."""
    start, wait = clock(), FIRST_WAIT
    while True:
        try:
            return fn()
        except Exception as err:
            if permanent(err) or clock() - start + wait > GIVE_UP_AFTER:
                raise
            name = type(err).__name__
            log.warning("download failed (%s: %s); retrying in %.0f s", name, err, wait)
            sleep(wait)
            wait = min(2 * wait, MAX_WAIT)
