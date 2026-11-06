import httpx
import pytest

from joblens.collect import collect_arbeitnow, collect_ba, read_raw, write_jsonl
from joblens.http import PoliteClient, RetryableHTTPError
from joblens.sources import BA_HEADERS, ba_encode, parse_arbeitnow_page, parse_ba_detail


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.t += s


def make_client(tmp_path, handler, **kw):
    clk = FakeClock()
    c = PoliteClient(
        tmp_path / "cache",
        transport=httpx.MockTransport(handler),
        sleep=clk.sleep,
        clock=clk.now,
        **kw,
    )
    return c, clk


def test_rate_limit_and_cache(tmp_path):
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(str(req.url))
        assert "JobLensExtract" in req.headers["user-agent"]
        return httpx.Response(200, json={"ok": len(calls)})

    c, clk = make_client(tmp_path, handler, min_interval=1.0)
    assert c.get_json("https://x.test/a", {"p": 1}) == {"ok": 1}
    assert c.get_json("https://x.test/b") == {"ok": 2}
    assert clk.sleeps == [1.0]  # second request waited for the interval
    assert c.get_json("https://x.test/a", {"p": 1}) == {"ok": 1}  # cached
    assert c.network_calls == 2 and c.cache_hits == 1
    c.close()


def test_retry_on_429_then_success(tmp_path):
    n = {"i": 0}

    def handler(req):
        n["i"] += 1
        return httpx.Response(429) if n["i"] < 3 else httpx.Response(200, json=[1])

    c, clk = make_client(tmp_path, handler, min_interval=0)
    assert c.get_json("https://x.test/r") == [1]
    assert n["i"] == 3
    assert len(clk.sleeps) == 2 and all(0 <= s <= 60 for s in clk.sleeps)


def test_retry_gives_up(tmp_path):
    c, _ = make_client(tmp_path, lambda r: httpx.Response(503), min_interval=0, max_attempts=2)
    with pytest.raises(RetryableHTTPError):
        c.get_json("https://x.test/down")


def test_404_not_retried(tmp_path):
    n = {"i": 0}

    def handler(req):
        n["i"] += 1
        return httpx.Response(404)

    c, _ = make_client(tmp_path, handler, min_interval=0)
    with pytest.raises(httpx.HTTPStatusError):
        c.get_json("https://x.test/missing")
    assert n["i"] == 1


def test_parse_arbeitnow(fixture_json):
    rows = parse_arbeitnow_page(fixture_json("arbeitnow_page.json"))
    assert len(rows) == 3  # posting without description skipped
    assert rows[0].source == "arbeitnow" and rows[0].company == "Acme Analytics GmbH"


def test_ba_encode_matches_docs():
    assert ba_encode("10001-1002716922-S") == "MTAwMDEtMTAwMjcxNjkyMi1T"


def test_parse_ba_detail(fixture_json):
    p = parse_ba_detail(fixture_json("ba_detail.json"), "Softwareentwickler")
    assert p is not None and p.source == "bundesagentur"
    assert p.source_id == "10000-SYNTH0-S" and "65.000" in p.raw_text
    assert parse_ba_detail({"referenznummer": "x"}, "q") is None


def test_collect_end_to_end(tmp_path, fixture_json):
    search, detail, page = (
        fixture_json("ba_search.json"),
        fixture_json("ba_detail.json"),
        fixture_json("arbeitnow_page.json"),
    )

    def handler(req: httpx.Request):
        u = str(req.url)
        if "job-board-api" in u:
            return httpx.Response(200, json=page)
        if "/pc/v6/jobs" in u:
            assert req.headers["x-api-key"] == "jobboerse-jobsuche"
            return httpx.Response(200, json=search)
        if "jobdetails" in u:
            if u.endswith(ba_encode("10000-SYNTH1-S")):
                return httpx.Response(404)
            return httpx.Response(200, json=detail)
        return httpx.Response(500)

    c, _ = make_client(tmp_path, handler, min_interval=0, headers=BA_HEADERS)
    an = collect_arbeitnow(c, target=10)
    assert len(an) == 3
    ba = collect_ba(c, target=5, queries=[("Software", 1)], cities=["Berlin"])
    assert len(ba) == 1  # one detail 404s
    n = write_jsonl(tmp_path / "out.jsonl", an + ba)
    assert n == 4 and len(read_raw(tmp_path / "out.jsonl")) == 4


def test_collect_ba_stops_on_failures(tmp_path, fixture_json):
    search = fixture_json("ba_search.json")

    def handler(req):
        if "/pc/v6/jobs" in str(req.url):
            return httpx.Response(200, json=search)
        return httpx.Response(404)

    c, _ = make_client(tmp_path, handler, min_interval=0)
    with pytest.raises(RuntimeError):
        collect_ba(c, 5, queries=[("x", 1)], cities=["y"], max_consecutive_failures=2)
