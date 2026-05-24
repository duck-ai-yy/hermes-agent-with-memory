"""The `web_fetch` tool — GET a URL, return text body (with SSRF guard).

v0.9 / M2 design pins:
  - GET only (POST/PUT come back when a use-case asks)
  - http / https scheme only
  - SSRF preflight: resolve host, refuse private / loopback / link-local /
    multicast / reserved / unspecified. Re-check after each redirect.
  - 15s timeout, 256 KB body cap
  - Content-Type must be text/* or application/json or application/*+json
  - Known TOCTOU between getaddrinfo() and connect: NOT fixed in v0.9
    (documented in design §1 / Spike 3, ratchet pin E5).
"""

from __future__ import annotations

import ipaddress
import socket
import time

import httpx

from .. import paths
from ..trace import events
from .registry import ToolResult, tool

WEB_FETCH_TIMEOUT = 15.0
MAX_FETCH_BYTES = 256 * 1024
USER_AGENT = "mneme/0.9 (+local)"
MAX_REDIRECTS = 10


def _classify_ip(addr: str) -> str | None:
    """Return a reason string if `addr` is in any blocked range; else None."""
    ip = ipaddress.ip_address(addr)
    if ip.is_private:
        return "private address"
    if ip.is_loopback:
        return "loopback address"
    if ip.is_link_local:
        return "link-local address"
    if ip.is_multicast:
        return "multicast address"
    if ip.is_reserved:
        return "reserved address"
    if ip.is_unspecified:
        return "unspecified address"
    return None


def _ssrf_check(host: str) -> tuple[bool, str]:
    """Resolve `host` and refuse if any address is blocked.

    Returns (ok, reason_or_empty). DNS failure -> (False, "DNS failure: ...").
    """
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        return False, f"DNS failure: {exc}"
    for info in infos:
        addr = info[4][0]
        reason = _classify_ip(addr)
        if reason is not None:
            return False, reason
    return True, ""


def _audit_event(url: str, host: str) -> None:
    """Best-effort audit log of an outbound request. Never raises."""
    try:
        ep = getattr(paths, "EVENTS_PATH", None)
        if ep is not None and ep.exists():
            events.append(ep, kind="audit", endpoint="web_fetch",
                          url=url, host=host)
    except Exception:
        pass


def _content_type_ok(ct: str) -> bool:
    """text/*, application/json, application/*+json — see design §4.2."""
    ct = ct.split(";", 1)[0].strip().lower()
    if ct.startswith("text/"):
        return True
    if ct == "application/json":
        return True
    if ct.startswith("application/") and ct.endswith("+json"):
        return True
    return False


@tool
def web_fetch(url: str, max_bytes: int = 65536) -> ToolResult:
    """Fetch a public web page or JSON API by GET. Returns the response body as text (truncated if large). Only http and https schemes are allowed; private, loopback, and link-local hosts are refused; non-text content types are refused.

    Args:
        url: Absolute URL to fetch (http:// or https://).
        max_bytes: Maximum bytes of the body to return (capped at 256 KB).
    """  # noqa: E501
    start = time.time()
    cap = min(max_bytes, MAX_FETCH_BYTES)

    # Parse the URL to get scheme + host for the SSRF preflight.
    try:
        parsed = httpx.URL(url)
    except Exception as exc:
        msg = f"WebFetchError: {type(exc).__name__}: {exc}"
        return ToolResult(content=msg, is_error=True,
                          audit={"url": url, "error": type(exc).__name__})

    scheme = (parsed.scheme or "").lower()
    if scheme not in ("http", "https"):
        msg = f"WebFetchError: scheme not allowed: {scheme}"
        return ToolResult(content=msg, is_error=True,
                          audit={"url": url, "error": "SchemeNotAllowed"})

    host = parsed.host
    if not host:
        msg = "WebFetchError: missing host"
        return ToolResult(content=msg, is_error=True,
                          audit={"url": url, "error": "MissingHost"})

    ok, reason = _ssrf_check(host)
    if not ok:
        if reason.startswith("DNS failure:"):
            msg = f"WebFetchError: {reason}"
            return ToolResult(content=msg, is_error=True,
                              audit={"url": url, "error": "DNSFailure"})
        msg = f"WebFetchError: SSRF blocked: {reason}"
        return ToolResult(content=msg, is_error=True,
                          audit={"url": url, "error": "SSRFBlocked"})

    # Preflight passed — log the outbound intent BEFORE opening the connection.
    _audit_event(url, host)

    # Manually follow redirects so we can re-SSRF each hop. httpx's auto-follow
    # doesn't expose intermediate hosts before connecting (Spike 3).
    current_url = url
    current_host = host
    try:
        with httpx.Client(
            timeout=WEB_FETCH_TIMEOUT,
            follow_redirects=False,
            headers={"User-Agent": USER_AGENT},
        ) as client:
            for _hop in range(MAX_REDIRECTS + 1):
                resp = client.get(current_url)
                # Redirect? Re-SSRF the new URL, then loop.
                if resp.is_redirect:
                    loc = resp.headers.get("location")
                    if not loc:
                        # Some servers send 3xx without a Location — treat
                        # the body we have as the result.
                        break
                    # Resolve relative locations against the current URL.
                    next_url = str(httpx.URL(current_url).join(loc))
                    next_parsed = httpx.URL(next_url)
                    next_scheme = (next_parsed.scheme or "").lower()
                    if next_scheme not in ("http", "https"):
                        msg = f"WebFetchError: scheme not allowed: {next_scheme}"
                        return ToolResult(content=msg, is_error=True,
                                          audit={"url": next_url,
                                                 "error": "SchemeNotAllowed"})
                    next_host = next_parsed.host
                    if not next_host:
                        msg = "WebFetchError: missing host"
                        return ToolResult(content=msg, is_error=True,
                                          audit={"url": next_url,
                                                 "error": "MissingHost"})
                    ok2, reason2 = _ssrf_check(next_host)
                    if not ok2:
                        if reason2.startswith("DNS failure:"):
                            msg = f"WebFetchError: {reason2}"
                            return ToolResult(content=msg, is_error=True,
                                              audit={"url": next_url,
                                                     "error": "DNSFailure"})
                        msg = f"WebFetchError: SSRF blocked: {reason2}"
                        return ToolResult(content=msg, is_error=True,
                                          audit={"url": next_url,
                                                 "error": "SSRFBlocked"})
                    current_url = next_url
                    current_host = next_host
                    continue
                # Not a redirect — this is the final response.
                break
            else:
                # for/else: loop exhausted without break.
                msg = f"WebFetchError: too many redirects (>{MAX_REDIRECTS})"
                return ToolResult(content=msg, is_error=True,
                                  audit={"url": current_url,
                                         "error": "TooManyRedirects"})
    except httpx.TimeoutException:
        msg = "WebFetchError: timeout after 15s"
        return ToolResult(content=msg, is_error=True,
                          audit={"url": current_url, "error": "Timeout"})
    except Exception as exc:
        msg = f"WebFetchError: {type(exc).__name__}: {exc}"
        return ToolResult(content=msg, is_error=True,
                          audit={"url": current_url, "error": type(exc).__name__})

    duration_ms = int((time.time() - start) * 1000)
    status = resp.status_code

    # Content-Type gate (only on the final response).
    ct = resp.headers.get("content-type", "")
    if not _content_type_ok(ct):
        msg = f"WebFetchError: binary content refused: {ct}"
        return ToolResult(content=msg, is_error=True,
                          audit={"url": current_url, "host": current_host,
                                 "status": status, "duration_ms": duration_ms,
                                 "error": "BinaryContent"})

    if status >= 400:
        msg = f"WebFetchError: HTTP {status}"
        return ToolResult(content=msg, is_error=True,
                          audit={"url": current_url, "host": current_host,
                                 "status": status, "duration_ms": duration_ms,
                                 "error": "HTTPError"})

    raw = resp.content or b""
    total = len(raw)
    truncated = total > cap
    body = raw[:cap].decode("utf-8", errors="replace")
    if truncated:
        # Design pins this marker as a literal (262144 = MAX_FETCH_BYTES);
        # the user's max_bytes is enforced silently inside cap.
        body += "\n... [body truncated, capped at 262144 bytes]"

    return ToolResult(
        content=body,
        is_error=False,
        audit={
            "url": current_url,
            "host": current_host,
            "status": status,
            "bytes": min(total, cap),
            "duration_ms": duration_ms,
            "truncated": truncated,
        },
    )
