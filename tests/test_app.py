"""Smoke tests for the viewer. Skipped whole if no run exists on this machine —
the app is a window onto artifacts/, and there is nothing to look at without them."""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from cmr import app as appmod  # noqa: E402

client = TestClient(appmod.app)
RUNS = client.get("/api/runs").json()
needs_run = pytest.mark.skipif(not RUNS, reason="no artifacts/runs/* on this machine")


def _biggest() -> dict:
    return max(RUNS, key=lambda r: r["n_subjects"])


def test_index_is_self_contained() -> None:
    r = client.get("/")
    assert r.status_code == 200
    # The viva room may have no internet: nothing may be fetched from a CDN.
    assert "http://" not in r.text.replace("http://127.0.0.1", "").replace("http://localhost", "")


@needs_run
def test_subject_list_returns_rows() -> None:
    r = client.get("/api/subjects", params={"run": _biggest()["config_id"]})
    assert r.status_code == 200
    d = r.json()
    assert d["n"] == len(d["rows"]) > 0
    assert {"subject_id", "status", "refused", "numeric_fidelity"} <= set(d["rows"][0])


@needs_run
def test_subject_detail_returns_a_report() -> None:
    cid = _biggest()["config_id"]
    rows = client.get("/api/subjects", params={"run": cid}).json()["rows"]
    sid = next(r["subject_id"] for r in rows if not r["refused"])
    d = client.get(f"/api/subject/{sid}", params={"run": cid}).json()
    assert d["subject_id"] == sid
    assert d["report"]["diagnosis"]
    assert d["trace"] and d["trace"][0]["node"] == "segment"


@needs_run
def test_refused_subject_shows_refused_and_has_no_report() -> None:
    """H3, asserted: the refusal must survive the round trip to the browser."""
    cid = _biggest()["config_id"]
    rows = client.get("/api/subjects", params={"run": cid}).json()["rows"]
    refused = [r for r in rows if r["refused"]]
    if not refused:
        pytest.skip(f"{cid} has no refusals")
    d = client.get(f"/api/subject/{refused[0]['subject_id']}", params={"run": cid}).json()
    assert d["refused"] is True
    assert d["status"] == "failed_segmentation"
    assert d["report"] is None  # Agent 3 did NOT diagnose
    assert any(e["node"] == "refuse" for e in d["trace"])


def test_corpus_search_returns_a_chunk() -> None:
    if not appmod.corpus():
        pytest.skip("no corpus on this machine")
    d = client.get("/api/chunks", params={"q": "ejection fraction"}).json()
    assert d["n_hits"] > 0
    c = d["chunks"][0]
    assert client.get(f"/api/chunk/{c['chunk_id']}").json()["text"] == c["text"]


def test_unknown_chunk_404s_rather_than_inventing_one() -> None:
    assert client.get("/api/chunk/deadbeefdeadbeef").status_code == 404


def test_unknown_run_404s() -> None:
    assert client.get("/api/subjects", params={"run": "no_such_run"}).status_code == 404
