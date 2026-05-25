"""v0.10 HTTP /chat session_id behavior — F-section + E-section (HTTP variant)
+ R-12.

Lead must-fix #1: this file uses a real TestClient (via the http_client
fixture in conftest.py). The legacy tests in test_server_chat.py bypass the
HTTP boundary (per B22 lesson, that's correct for the no-tools invariant).
But v0.10's session_id contract lives ON the boundary — the
ChatRequest.session_id field, the response body's session_id echo, the
"None or whitespace -> mint new" branch — so the test must exercise the
real ASGI app.

Architect §7 (HTTP path):
  - request.session_id None       -> mint a new ULID, return it in response
  - request.session_id ""         -> same as None: mint new
  - request.session_id "   "      -> same as None: mint new (whitespace-only)
  - request.session_id "x"        -> echo "x" back; do NOT validate against DB

Response field order is pinned by design §7: text, trace_id,
citation_quality, session_id.
"""

from __future__ import annotations

import json
import re


# -- helpers ----------------------------------------------------------------


def _events(path):
    return [
        json.loads(line) for line in
        path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _is_ulid(s: str) -> bool:
    return bool(re.fullmatch(r"[0-9A-HJKMNP-TV-Z]{26}", s))


# =============================================================================
# §F — HTTP-layer session_id end-to-end
# =============================================================================


def test_F1_chat_with_no_session_id_mints_new_ulid_in_response(http_client):
    """F1: POST /chat with session_id omitted. Response body must contain
    a freshly-minted 26-char ULID under the `session_id` key."""
    resp = http_client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "hello"}]},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "session_id" in body
    assert _is_ulid(body["session_id"]), (
        f"server did not mint a real ULID: {body['session_id']!r}"
    )


def test_F2_chat_with_caller_session_id_echoes_verbatim(http_client):
    """F2: a caller-supplied session_id round-trips byte-for-byte. The
    server is stateless — no normalization, no rejection."""
    sess = "client-managed-session-99"
    resp = http_client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "hi"}],
              "session_id": sess},
    )
    assert resp.status_code == 200
    assert resp.json()["session_id"] == sess


def test_F3_two_chat_calls_same_caller_session_id_share_session_in_db(
    http_client, tmp_path,
):
    """F3: two HTTP /chat calls with the same caller-supplied session_id
    write slices that all share that session_id in the DB. End-to-end
    proof that the boundary forwards session_id down to ingest."""
    sess = "client-managed-multi-turn"
    for content in ("first", "second"):
        resp = http_client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": content}],
                  "session_id": sess},
        )
        assert resp.status_code == 200, resp.text

    from mneme.memory import store
    from mneme import paths
    cx = store.connect(paths.DB_PATH)
    sessions = {
        r[0] for r in cx.execute("SELECT DISTINCT session_id FROM slices")
    }
    cx.close()
    assert sessions == {sess}, (
        f"server did not pin session_id end-to-end; sessions on disk: {sessions}"
    )


def test_F4_response_field_order_text_trace_id_citation_quality_session_id(
    http_client,
):
    """F4 (design §7): the response body's key order is pinned —
    text, trace_id, citation_quality, session_id. FastAPI serializes the
    dict directly, so insertion order is the on-the-wire order.

    We parse the raw JSON text (resp.text, not .json()) to preserve order."""
    resp = http_client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    # Order assertion on the raw JSON text.
    raw = resp.text
    expected_keys = ["text", "trace_id", "citation_quality", "session_id"]
    positions = [raw.find(f'"{k}"') for k in expected_keys]
    assert all(p > 0 for p in positions), (
        f"response missing one of {expected_keys}: positions={positions}, "
        f"raw={raw!r}"
    )
    # Strictly increasing positions == correct key order.
    assert positions == sorted(positions), (
        f"response field order broken; key positions = "
        f"{dict(zip(expected_keys, positions))} in {raw!r}"
    )


def test_F5_http_chat_writes_session_id_into_trace_events(
    http_client, tmp_path,
):
    """F5: the HTTP path's trace events on disk carry session_id == the
    one the server sent back. Ties the response body to the audit log."""
    sess = "http-traceme"
    resp = http_client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "audit me"}],
              "session_id": sess},
    )
    assert resp.status_code == 200
    trace_id = resp.json()["trace_id"]

    from mneme import paths
    ev = _events(paths.EVENTS_PATH)
    trace_rows = [r for r in ev
                  if r.get("kind") == "trace" and r.get("id") == trace_id]
    assert len(trace_rows) == 2  # open + close
    for r in trace_rows:
        assert r["session_id"] == sess, r


# =============================================================================
# §E (HTTP variant) — must-not behaviors on the server boundary
# =============================================================================


def test_E6_http_request_with_unknown_session_id_does_not_raise(http_client):
    """E6 (HTTP variant): unknown session_id on /chat returns 200 — the
    server is stateless about sessions, so unknown ids are silently
    accepted. (E6's CLI variant lives in test_session_negative.py.)"""
    resp = http_client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "hi"}],
              "session_id": "totally-unknown-id-99999"},
    )
    assert resp.status_code == 200
    assert resp.json()["session_id"] == "totally-unknown-id-99999"


# =============================================================================
# §R — R-12: server response_id IS the value used on disk
# =============================================================================


def test_R_12_response_session_id_matches_session_id_written_to_slices(
    http_client,
):
    """R-12: the session_id field in the JSON response body MUST equal the
    session_id column written to slices for that turn. Catches the silent
    bug where the server mints one id for the response but uses a
    different one (e.g. turn_id) for ingest."""
    resp = http_client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "r12 test"}]},
    )
    assert resp.status_code == 200
    response_sess = resp.json()["session_id"]

    from mneme.memory import store
    from mneme import paths
    cx = store.connect(paths.DB_PATH)
    db_sessions = {
        r[0] for r in cx.execute("SELECT DISTINCT session_id FROM slices")
    }
    cx.close()
    assert response_sess in db_sessions, (
        f"response.session_id={response_sess!r} not found in DB sessions "
        f"{db_sessions}"
    )
    # And nothing else snuck in (a turn_id-as-session leak would create
    # a different distinct value).
    assert db_sessions == {response_sess}, (
        f"extra sessions in DB beyond response.session_id: "
        f"{db_sessions - {response_sess}}"
    )


def test_R_12_minted_session_id_is_well_formed_ulid(http_client):
    """R-12 (continued): the minted id is a real 26-char Crockford-base32
    ULID. Pin format so a future refactor to e.g. uuid4 doesn't slip past."""
    resp = http_client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "x"}]},
    )
    body_sess = resp.json()["session_id"]
    assert _is_ulid(body_sess), (
        f"minted session_id is not a ULID: {body_sess!r}"
    )
