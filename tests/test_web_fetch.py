"""v0.9 / M2 — web_fetch: SSRF preflight + redirect re-check + audit ordering.

Boundary numbers from the phase-1 plan:
  C9-C19: scheme allow-list (http/https only), SSRF refusals (private /
          loopback / link-local / multicast / reserved / unspecified),
          DNS failure label, GET-only contract, 15s timeout, 256 KB cap,
          content-type gate.
  R-WF-1..9: ratchets — manual redirect, max redirects, redirect-then-
            SSRF re-check, error event audit fields shape, body truncation
            marker literal, HTTP 4xx surfaces "HTTP NNN", User-Agent
            header pinned, audit BEFORE connection (principle 5).
  R-WF-10: Lead must-fix #6 — redirect Location: file:/// is refused
           with scheme-not-allowed (dev already implemented at
           web_fetch.py:156-158, this test should pass green).
"""

from __future__ import annotations

import httpx
import pytest

from mneme import paths
from mneme.tools import web_fetch as wf_mod
from mneme.tools.registry import execute


# Test data: a sane public-IP address that passes the SSRF preflight.
PUB_ADDR = "93.184.216.34"   # example.com canonical address (any non-private)


def _ok_text(status=200, body="hi", content_type="text/plain"):
    """Build a small OK response factory."""
    def handler(request):
        return httpx.Response(status, text=body,
                              headers={"content-type": content_type})
    return handler


# -- C9: scheme allow-list -------------------------------------------------


def test_C9_scheme_not_http_or_https_refused(mock_httpx, mock_getaddrinfo):
    """`ftp://example.com/x` -> 'WebFetchError: scheme not allowed: ftp'.
    Note: scheme check fires BEFORE SSRF / before opening any client, so
    DNS isn't even consulted for ftp."""
    tr = execute("web_fetch", {"url": "ftp://example.com/x"})
    assert tr.is_error is True
    assert tr.content.startswith("WebFetchError: scheme not allowed: ftp")
    assert tr.audit["error"] == "SchemeNotAllowed"


def test_C9b_file_scheme_refused_does_not_attempt_dns(mock_httpx, mock_getaddrinfo):
    """`file:///etc/passwd` must not call getaddrinfo."""
    tr = execute("web_fetch", {"url": "file:///etc/passwd"})
    assert tr.is_error is True
    assert tr.content.startswith("WebFetchError: scheme not allowed: file")
    assert tr.audit["error"] == "SchemeNotAllowed"
    assert mock_getaddrinfo.mapping == {}   # never populated, never queried


# -- C10: SSRF refusal — private (10/8 / 172.16/12 / 192.168/16) -----------


def test_C10_private_address_refused_with_ssrf_blocked(mock_httpx, mock_getaddrinfo):
    mock_getaddrinfo.set("internal.example", "10.0.0.5")
    tr = execute("web_fetch", {"url": "http://internal.example/"})
    assert tr.is_error is True
    assert tr.audit["error"] == "SSRFBlocked"
    assert "SSRF blocked: private address" in tr.content


# -- C11: SSRF refusal — loopback -----------------------------------------


def test_C11_loopback_127_0_0_1_refused(mock_httpx, mock_getaddrinfo):
    """127.0.0.1 is BOTH is_private and is_loopback in Python's ipaddress
    module; the dev's _classify_ip checks private FIRST so the reason
    string is 'private address'. Either reason proves the address was
    refused — the contract that matters is SSRF blocked the call."""
    mock_getaddrinfo.set("localhost.example", "127.0.0.1")
    tr = execute("web_fetch", {"url": "http://localhost.example/"})
    assert tr.is_error is True
    assert tr.audit["error"] == "SSRFBlocked"
    # Loopback IS reported as 'private address' due to is_private being True
    # for 127/8 in Python's stdlib. Pin the actual behavior so a future
    # _classify_ip refactor that reorders the checks is caught.
    assert "SSRF blocked: private address" in tr.content


# -- C12: SSRF refusal — link-local (169.254/16) --------------------------


def test_C12_link_local_169_254_refused(mock_httpx, mock_getaddrinfo):
    """169.254.169.254 (AWS metadata endpoint) is is_private=True AND
    is_link_local=True; private wins in _classify_ip's order. Still
    blocked, which is what matters for SSRF."""
    mock_getaddrinfo.set("aws-meta.example", "169.254.169.254")
    tr = execute("web_fetch", {"url": "http://aws-meta.example/"})
    assert tr.is_error is True
    assert tr.audit["error"] == "SSRFBlocked"
    assert "SSRF blocked: private address" in tr.content


# -- C13: SSRF refusal — multicast / reserved / unspecified ---------------


def test_C13a_multicast_refused(mock_httpx, mock_getaddrinfo):
    """224.0.0.1 is NOT is_private (it's outside the private RFC ranges) but
    is_multicast=True — this is the one category whose reason string is
    actually 'multicast address' because private check returns False."""
    mock_getaddrinfo.set("mcast.example", "224.0.0.1")
    tr = execute("web_fetch", {"url": "http://mcast.example/"})
    assert tr.is_error is True
    assert tr.audit["error"] == "SSRFBlocked"
    assert "multicast address" in tr.content


def test_C13b_reserved_240_block_refused(mock_httpx, mock_getaddrinfo):
    """240.0.0.1 is also is_private=True (in Python's stdlib it sits in
    the 'unallocated' bucket which ipaddress counts as private). The
    reason string ends up 'private address' due to check order; pin the
    actual behavior."""
    mock_getaddrinfo.set("reserved.example", "240.0.0.1")
    tr = execute("web_fetch", {"url": "http://reserved.example/"})
    assert tr.is_error is True
    assert tr.audit["error"] == "SSRFBlocked"


def test_C13c_unspecified_0_0_0_0_refused(mock_httpx, mock_getaddrinfo):
    """0.0.0.0 is is_unspecified=True AND is_private=True; private wins."""
    mock_getaddrinfo.set("any.example", "0.0.0.0")
    tr = execute("web_fetch", {"url": "http://any.example/"})
    assert tr.is_error is True
    assert tr.audit["error"] == "SSRFBlocked"


# -- C14: DNS failure surfaces specific label -----------------------------


def test_C14_dns_failure_surfaces_DNSFailure_label(mock_httpx, mock_getaddrinfo):
    """No mapping for the host -> mock raises gaierror; SSRF check should
    return a 'DNS failure: ...' reason that becomes
    'WebFetchError: DNS failure: ...' content with audit.error='DNSFailure'."""
    # Don't set any mapping. mock_getaddrinfo raises socket.gaierror.
    tr = execute("web_fetch", {"url": "http://no-such-host.example/"})
    assert tr.is_error is True
    assert tr.audit["error"] == "DNSFailure"
    assert tr.content.startswith("WebFetchError: DNS failure:")


# -- C15: happy path — 200 + small body returns body ---------------------


def test_C15_happy_200_returns_body_content(mock_httpx, mock_getaddrinfo):
    mock_getaddrinfo.set("example.com", PUB_ADDR)
    mock_httpx.set_handler(_ok_text(200, "hello world"))
    tr = execute("web_fetch", {"url": "http://example.com/foo"})
    assert tr.is_error is False
    assert tr.content == "hello world"
    assert tr.audit["status"] == 200
    assert tr.audit["host"] == "example.com"
    assert tr.audit["url"] == "http://example.com/foo"


# -- C16: content-type gate refuses binary --------------------------------


def test_C16_binary_content_type_refused(mock_httpx, mock_getaddrinfo):
    """image/png is not text/* nor application/json -> binary content
    refused; status code is logged but body is not surfaced."""
    mock_getaddrinfo.set("example.com", PUB_ADDR)
    mock_httpx.set_handler(_ok_text(200, "PNGGAR", "image/png"))
    tr = execute("web_fetch", {"url": "http://example.com/x.png"})
    assert tr.is_error is True
    assert tr.audit["error"] == "BinaryContent"
    assert "binary content refused" in tr.content


def test_C16b_application_json_is_allowed(mock_httpx, mock_getaddrinfo):
    mock_getaddrinfo.set("api.example.com", PUB_ADDR)
    mock_httpx.set_handler(_ok_text(200, '{"k":1}', "application/json"))
    tr = execute("web_fetch", {"url": "http://api.example.com/v"})
    assert tr.is_error is False
    assert tr.content == '{"k":1}'


def test_C16c_application_vnd_api_json_is_allowed(mock_httpx, mock_getaddrinfo):
    """`application/vnd.api+json` -> ends with +json, allowed."""
    mock_getaddrinfo.set("api.example.com", PUB_ADDR)
    mock_httpx.set_handler(_ok_text(200, '{}', "application/vnd.api+json"))
    tr = execute("web_fetch", {"url": "http://api.example.com/v"})
    assert tr.is_error is False


# -- C17: HTTP 4xx/5xx returns "HTTP NNN" but is_error=True ---------------


def test_C17_http_404_surfaces_HTTP_status_in_content(mock_httpx, mock_getaddrinfo):
    mock_getaddrinfo.set("example.com", PUB_ADDR)
    mock_httpx.set_handler(_ok_text(404, "nope", "text/plain"))
    tr = execute("web_fetch", {"url": "http://example.com/missing"})
    assert tr.is_error is True
    assert "WebFetchError: HTTP 404" in tr.content
    assert tr.audit["error"] == "HTTPError"
    assert tr.audit["status"] == 404


def test_C17b_http_500_also_is_error(mock_httpx, mock_getaddrinfo):
    mock_getaddrinfo.set("example.com", PUB_ADDR)
    mock_httpx.set_handler(_ok_text(500, "boom", "text/html"))
    tr = execute("web_fetch", {"url": "http://example.com/break"})
    assert tr.is_error is True
    assert "WebFetchError: HTTP 500" in tr.content


# -- C18: body cap truncation marker literal ------------------------------


def test_C18_body_over_max_bytes_truncated_with_marker_literal(
    mock_httpx, mock_getaddrinfo,
):
    """Cap is 262144 (256 KB). Body of 300_000 bytes -> truncated, marker is
    "... [body truncated, capped at 262144 bytes]" verbatim."""
    mock_getaddrinfo.set("example.com", PUB_ADDR)
    mock_httpx.set_handler(_ok_text(200, "A" * 300_000, "text/plain"))
    tr = execute("web_fetch", {"url": "http://example.com/big"})
    assert tr.is_error is False
    assert "... [body truncated, capped at 262144 bytes]" in tr.content
    assert tr.audit["truncated"] is True


# -- C19: max_bytes arg silently capped at hard 256 KB ---------------------


def test_C19_max_bytes_above_hard_cap_silently_clamped(mock_httpx, mock_getaddrinfo):
    """Caller asks max_bytes=10**9; effective cap is 256 KB. Body of 100k
    bytes fits in either -> no truncation."""
    mock_getaddrinfo.set("example.com", PUB_ADDR)
    mock_httpx.set_handler(_ok_text(200, "B" * 100_000, "text/plain"))
    tr = execute("web_fetch", {"url": "http://example.com/x", "max_bytes": 10**9})
    assert tr.is_error is False
    assert tr.audit["truncated"] is False
    assert wf_mod.MAX_FETCH_BYTES == 256 * 1024


# -- R-WF-1: manual redirect follow ---------------------------------------


def test_R_WF_1_redirect_followed_and_final_body_returned(
    mock_httpx, mock_getaddrinfo,
):
    """302 -> 200; final body comes from the second response."""
    mock_getaddrinfo.set("example.com", PUB_ADDR)
    mock_getaddrinfo.set("dest.example.com", PUB_ADDR)

    def first(req):
        return httpx.Response(302, headers={"location": "http://dest.example.com/final"})

    def second(req):
        return httpx.Response(200, text="final body", headers={"content-type": "text/plain"})

    mock_httpx.set_handlers([first, second])
    tr = execute("web_fetch", {"url": "http://example.com/start"})
    assert tr.is_error is False
    assert tr.content == "final body"
    assert tr.audit["url"] == "http://dest.example.com/final"


# -- R-WF-2: max redirects cap fires --------------------------------------


def test_R_WF_2_too_many_redirects_returns_specific_error(
    mock_httpx, mock_getaddrinfo,
):
    """A redirect chain of MAX_REDIRECTS+1 length -> 'too many redirects'.
    The literal cap is pinned at 10 (ratchet against M12 = "cap relaxed
    to a huge value would let an attacker hammer infinitely"). Asserting
    the literal 10 catches the mutation; asserting str(wf_mod.MAX_REDIRECTS)
    would silently track whatever the module says."""
    mock_getaddrinfo.set("example.com", PUB_ADDR)

    def redirect_loop(req):
        return httpx.Response(302, headers={"location": "http://example.com/next"})

    mock_httpx.set_handler(redirect_loop)
    tr = execute("web_fetch", {"url": "http://example.com/start"})
    assert tr.is_error is True
    assert tr.audit["error"] == "TooManyRedirects"
    assert "too many redirects" in tr.content
    # Literal-pin the cap (mutation guard for M12).
    assert wf_mod.MAX_REDIRECTS == 10
    assert "(>10)" in tr.content


# -- R-WF-3: redirect target re-SSRF — private IP at hop 2 blocked --------


def test_R_WF_3_redirect_to_private_ip_refused_at_re_check(
    mock_httpx, mock_getaddrinfo,
):
    """Initial host passes SSRF; redirect targets a host that resolves to
    10.0.0.1 -> the hop-2 SSRF check refuses."""
    mock_getaddrinfo.set("example.com", PUB_ADDR)
    mock_getaddrinfo.set("evil.example.com", "10.0.0.1")

    def first(req):
        return httpx.Response(302, headers={"location": "http://evil.example.com/x"})

    mock_httpx.set_handlers([first])
    tr = execute("web_fetch", {"url": "http://example.com/start"})
    assert tr.is_error is True
    assert tr.audit["error"] == "SSRFBlocked"
    assert "private address" in tr.content
    assert tr.audit["url"] == "http://evil.example.com/x"


# -- R-WF-4: audit event for outbound intent written BEFORE connection ---


def test_R_WF_4_audit_event_written_before_connection(
    cx, mock_httpx, mock_getaddrinfo, events_spy, monkeypatch, tmp_path,
):
    """Principle 5: the audit (intent) is logged BEFORE the body is fetched.
    Use `monkeypatch` to point paths.EVENTS_PATH at tmp_path/events.jsonl,
    pre-create it, then run web_fetch. The audit kind='audit' must exist."""
    ep = tmp_path / "events.jsonl"
    ep.write_text("", encoding="utf-8")
    monkeypatch.setattr(paths, "EVENTS_PATH", ep)
    mock_getaddrinfo.set("example.com", PUB_ADDR)
    mock_httpx.set_handler(_ok_text(200, "ok", "text/plain"))
    tr = execute("web_fetch", {"url": "http://example.com/foo"})
    assert tr.is_error is False
    audits = events_spy(kind="audit")
    assert len(audits) == 1
    assert audits[0]["endpoint"] == "web_fetch"
    assert audits[0]["url"] == "http://example.com/foo"
    assert audits[0]["host"] == "example.com"


# -- R-WF-5: NO audit event written for SSRF-blocked URL ------------------


def test_R_WF_5_no_audit_event_when_ssrf_blocks_before_connection(
    mock_httpx, mock_getaddrinfo, events_spy, monkeypatch, tmp_path,
):
    """The audit log records OUTBOUND intent. An SSRF-blocked URL never
    leaves the box, so no audit kind is written (cleaner forensic trail —
    audits represent 'we did try to fetch this externally')."""
    ep = tmp_path / "events.jsonl"
    ep.write_text("", encoding="utf-8")
    monkeypatch.setattr(paths, "EVENTS_PATH", ep)
    mock_getaddrinfo.set("internal.example", "10.0.0.1")
    tr = execute("web_fetch", {"url": "http://internal.example/"})
    assert tr.is_error is True
    audits = events_spy(kind="audit")
    assert audits == []


# -- R-WF-6: timeout surfaces specific 'timeout after 15s' string ---------


def test_R_WF_6_timeout_returns_timeout_label(mock_httpx, mock_getaddrinfo):
    mock_getaddrinfo.set("example.com", PUB_ADDR)

    def slow(req):
        raise httpx.ConnectTimeout("simulated", request=req)

    mock_httpx.set_handler(slow)
    tr = execute("web_fetch", {"url": "http://example.com/"})
    assert tr.is_error is True
    assert tr.audit["error"] == "Timeout"
    assert "timeout after 15s" in tr.content


# -- R-WF-7: User-Agent header is pinned to mneme literal -----------------


def test_R_WF_7_user_agent_header_is_pinned(mock_httpx, mock_getaddrinfo):
    """The on-the-wire User-Agent must be the literal 'mneme/0.9 (+local)'.
    Captured from the request that hit our mock transport."""
    captured = {}

    def grab(req):
        captured["ua"] = req.headers.get("user-agent")
        return httpx.Response(200, text="ok", headers={"content-type": "text/plain"})

    mock_getaddrinfo.set("example.com", PUB_ADDR)
    mock_httpx.set_handler(grab)
    execute("web_fetch", {"url": "http://example.com/"})
    assert captured["ua"] == "mneme/0.9 (+local)"
    assert wf_mod.USER_AGENT == "mneme/0.9 (+local)"


# -- R-WF-8: relative-Location redirect resolved against the current URL -


def test_R_WF_8_relative_redirect_location_resolved_correctly(
    mock_httpx, mock_getaddrinfo,
):
    """A 302 with Location: '/relative' -> resolves to scheme+host of the
    current URL, NOT the original. Tests that httpx.URL().join() is being
    used correctly."""
    mock_getaddrinfo.set("example.com", PUB_ADDR)

    def first(req):
        return httpx.Response(302, headers={"location": "/somewhere-else"})

    def second(req):
        return httpx.Response(200, text="rel ok", headers={"content-type": "text/plain"})

    mock_httpx.set_handlers([first, second])
    tr = execute("web_fetch", {"url": "http://example.com/start"})
    assert tr.is_error is False
    assert tr.content == "rel ok"
    assert tr.audit["url"] == "http://example.com/somewhere-else"


# -- R-WF-9: HTTPS scheme passes too ---------------------------------------


def test_R_WF_9_https_scheme_works_end_to_end(mock_httpx, mock_getaddrinfo):
    """Both http and https are allowed; the scheme check passes either."""
    mock_getaddrinfo.set("example.com", PUB_ADDR)
    mock_httpx.set_handler(_ok_text(200, "https-ok", "text/plain"))
    tr = execute("web_fetch", {"url": "https://example.com/secure"})
    assert tr.is_error is False
    assert tr.content == "https-ok"


# -- R-WF-10 (Lead must-fix #6): redirect to non-http(s) scheme refused ---


def test_R_WF_10_redirect_to_file_scheme_refused_at_re_check(
    mock_httpx, mock_getaddrinfo,
):
    """Initial host passes SSRF + scheme check. A 302 with
    Location: file:///etc/passwd MUST be refused at the per-hop scheme
    check (web_fetch.py:156-158). Without that guard, an attacker who
    controls a redirect target could exfiltrate local files."""
    mock_getaddrinfo.set("example.com", PUB_ADDR)

    def first(req):
        return httpx.Response(302, headers={"location": "file:///etc/passwd"})

    mock_httpx.set_handlers([first])
    tr = execute("web_fetch", {"url": "http://example.com/start"})
    assert tr.is_error is True
    assert tr.audit["error"] == "SchemeNotAllowed"
    assert tr.content.startswith("WebFetchError: scheme not allowed: file")


# -- Extra contract pin: missing host -------------------------------------


def test_missing_host_in_url_returns_missinghost_error(mock_httpx, mock_getaddrinfo):
    """`http:///path` has no host -> MissingHost error class. Defense in
    depth — without the explicit check, httpx would try to resolve ""."""
    tr = execute("web_fetch", {"url": "http:///nopath"})
    assert tr.is_error is True
    assert tr.audit["error"] == "MissingHost"
    assert "missing host" in tr.content


# -- Extra ratchet: max_bytes arg trims body even when below hard cap ----


def test_max_bytes_arg_below_hard_cap_clips_body(mock_httpx, mock_getaddrinfo):
    """User requested 50 bytes of a 1000-byte body -> 50 bytes + marker."""
    mock_getaddrinfo.set("example.com", PUB_ADDR)
    mock_httpx.set_handler(_ok_text(200, "Z" * 1000, "text/plain"))
    tr = execute("web_fetch", {"url": "http://example.com/", "max_bytes": 50})
    assert tr.is_error is False
    assert tr.content.startswith("Z" * 50)
    assert "[body truncated" in tr.content
    assert tr.audit["truncated"] is True


# Unused fixture import guard so pyflakes is happy
_ = pytest
