import httpx2
import pytest
from huggingface_hub.errors import HfHubHTTPError

from tuncall.data.hub import GIVE_UP_AFTER, retrying


def http_error(status: int) -> HfHubHTTPError:
    request = httpx2.Request("GET", "https://huggingface.co/x")
    return HfHubHTTPError("x", response=httpx2.Response(status, request=request))


class Flaky:
    """Raises the given errors in turn, then returns "ok"; a fake clock advances on sleep."""

    def __init__(self, *errors: Exception) -> None:
        self.errors, self.now, self.waits = list(errors), 0.0, []

    def __call__(self) -> str:
        if self.errors:
            raise self.errors.pop(0)
        return "ok"

    def sleep(self, s: float) -> None:
        self.waits.append(s)
        self.now += s


def run(f: Flaky) -> str:
    return retrying(f, sleep=f.sleep, clock=lambda: f.now)


def test_network_errors_are_retried_with_growing_waits() -> None:
    f = Flaky(RuntimeError("CAS"), OSError("reset"), http_error(503), http_error(429), OSError())
    assert run(f) == "ok"
    assert f.waits == [30, 60, 120, 240, 480]


def test_waits_cap_at_ten_minutes_and_give_up_after_twelve_hours() -> None:
    f = Flaky(*[OSError()] * 1000)
    with pytest.raises(OSError):
        run(f)
    assert max(f.waits) == 600 and f.now <= GIVE_UP_AFTER < f.now + 600


@pytest.mark.parametrize("status", [401, 403, 404])
def test_permanent_hub_answers_are_raised_at_once(status: int) -> None:
    f = Flaky(http_error(status))
    with pytest.raises(HfHubHTTPError):
        run(f)
    assert f.waits == []
