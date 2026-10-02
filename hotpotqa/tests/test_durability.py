"""Tests for the durability guard (src/durability.py).

The guard is the mechanism standing between this project and a third data loss
(docs/WAYS_OF_WORKING.md "Durability rules"), so it is tested for both
directions: it must reject ephemeral paths, and it must not reject the
persistent ones -- a guard that cries wolf gets routed around.

Offline, no server, no corpus.
"""

import os
import subprocess
import sys

import pytest

from src import durability

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODULE = os.path.join(REPO, "hotpotqa", "src", "durability.py")


@pytest.fixture(autouse=True)
def _no_bypass(monkeypatch):
    """conftest sets the test bypass; these tests need the real behaviour."""
    monkeypatch.delenv(durability.ALLOW_ENV_VAR, raising=False)


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------

@pytest.mark.parametrize("path", [
    "/tmp/specmem/paperb",                 # the 2026-10-02 loss, verbatim
    "/tmp/specmem/paperb/pilot25_title_exact/pairs.jsonl",
    "/tmp/kilt_raw",
    "/dev/shm/x",
    "/var/tmp/y",
    "/run/z",
])
def test_known_ephemeral_paths_are_rejected(path):
    assert durability.is_durable(path) is False
    with pytest.raises(durability.EphemeralOutputError):
        durability.require_durable_outputs(out=path)


def test_home_is_durable():
    # $HOME is the JuiceFS mount that survived every recycle so far.
    assert durability.is_durable(os.path.expanduser("~/specmem-data/runs"))


def test_nonexistent_path_is_classified_by_its_nearest_existing_ancestor():
    # Output dirs do not exist yet when the guard runs -- that is the point.
    deep = os.path.expanduser("~/specmem-data/runs/does/not/exist/yet")
    assert not os.path.exists(deep)
    info = durability.classify(deep)
    assert info["durable"] is True
    assert info["probed"] == os.path.expanduser("~")  \
        or os.path.exists(info["probed"])

    deep_tmp = "/tmp/specmem/also/not/created/yet"
    assert durability.is_durable(deep_tmp) is False


def test_tmp_subpath_is_rejected_even_without_findmnt(monkeypatch):
    """The prefix rule must stand on its own.

    If findmnt is missing or /tmp happens to sit on a durable root, the path is
    still conventionally scratch and must not receive outputs.
    """
    monkeypatch.setattr(durability, "_findmnt", lambda p: (None, None))
    monkeypatch.setattr(durability, "_proc_mounts", lambda p: ("ext4", "/"))
    assert durability.is_durable("/tmp/specmem/out") is False


def test_overlay_mount_is_rejected_by_fstype(monkeypatch):
    """A path NOT under /tmp but on the overlay pod root is still ephemeral."""
    monkeypatch.setattr(durability, "_findmnt", lambda p: ("overlay", "/"))
    info = durability.classify("/opt/elsewhere/out")
    assert info["durable"] is False
    assert "overlay" in info["reason"]


def test_undeterminable_mount_is_assumed_durable(monkeypatch):
    """Fail open only when the mount is genuinely unknown, and say so."""
    monkeypatch.setattr(durability, "_findmnt", lambda p: (None, None))
    monkeypatch.setattr(durability, "_proc_mounts", lambda p: (None, None))
    info = durability.classify("/some/unknown/place")
    assert info["durable"] is True
    assert "could not be determined" in info["reason"]


# --------------------------------------------------------------------------
# require_durable_outputs
# --------------------------------------------------------------------------

def test_all_offending_paths_are_reported_not_just_the_first():
    with pytest.raises(durability.EphemeralOutputError) as exc:
        durability.require_durable_outputs(
            out="/tmp/a", cache="/dev/shm/b",
            good=os.path.expanduser("~/specmem-data/runs"))
    msg = str(exc.value)
    assert "/tmp/a" in msg and "/dev/shm/b" in msg
    assert "2 output path(s)" in msg
    # The durable one must not be blamed.
    assert "good =" not in msg


def test_error_names_the_offending_argument_and_points_at_the_rules():
    with pytest.raises(durability.EphemeralOutputError) as exc:
        durability.require_durable_outputs(criteria_scores="/tmp/x.jsonl")
    msg = str(exc.value)
    assert "criteria_scores" in msg
    assert "WAYS_OF_WORKING.md" in msg


def test_none_paths_are_skipped():
    # Optional outputs (e.g. --out defaulting to None) must not trip the guard.
    assert durability.require_durable_outputs(out=None) == {}


def test_returns_absolute_paths():
    resolved = durability.require_durable_outputs(
        out=os.path.expanduser("~/specmem-data/runs/."))
    assert os.path.isabs(resolved["out"])


def test_advice_names_spec_base_when_set(monkeypatch):
    monkeypatch.setenv("SPEC_BASE", "/home/someone/specmem-data")
    with pytest.raises(durability.EphemeralOutputError) as exc:
        durability.require_durable_outputs(out="/tmp/x")
    assert "/home/someone/specmem-data" in str(exc.value)


# --------------------------------------------------------------------------
# the test-only bypass
# --------------------------------------------------------------------------

def test_bypass_allows_ephemeral_but_warns(monkeypatch, capsys):
    monkeypatch.setenv(durability.ALLOW_ENV_VAR, "1")
    resolved = durability.require_durable_outputs(out="/tmp/fixture")
    assert resolved["out"] == "/tmp/fixture"
    err = capsys.readouterr().err
    assert "WARNING" in err
    assert "will NOT survive a pod recycle" in err


def test_bypass_requires_exactly_1(monkeypatch):
    monkeypatch.setenv(durability.ALLOW_ENV_VAR, "yes")
    with pytest.raises(durability.EphemeralOutputError):
        durability.require_durable_outputs(out="/tmp/fixture")


# --------------------------------------------------------------------------
# the CLI the shell entry scripts call
# --------------------------------------------------------------------------

def _run_cli(*paths, env=None):
    e = dict(os.environ)
    e.pop(durability.ALLOW_ENV_VAR, None)
    if env:
        e.update(env)
    return subprocess.run([sys.executable, MODULE, "--check", *paths],
                          capture_output=True, text=True, env=e)


def test_cli_exits_0_on_durable_paths():
    out = _run_cli(os.path.expanduser("~/specmem-data/runs"))
    assert out.returncode == 0


def test_cli_exits_1_and_explains_on_ephemeral_paths():
    out = _run_cli("/tmp/specmem/logs")
    assert out.returncode == 1
    assert "/tmp/specmem/logs" in out.stderr
    assert "refusing to run" in out.stderr


def test_cli_runs_under_system_python_without_the_pipeline_env():
    """The shell guard must work on a freshly recycled pod.

    scripts/env.sh calls this with the *system* python3, before setup_envs.sh
    has created $PIPELINE_PY, so the module must import with stdlib only.
    """
    out = subprocess.run(
        ["/usr/bin/python3", MODULE, "--check", "/tmp/x"],
        capture_output=True, text=True,
        env={k: v for k, v in os.environ.items()
             if k != durability.ALLOW_ENV_VAR})
    assert out.returncode == 1
    assert "refusing to run" in out.stderr


def test_quiet_flag_silences_success_output():
    p = os.path.expanduser("~/specmem-data/runs")
    e = {k: v for k, v in os.environ.items() if k != durability.ALLOW_ENV_VAR}
    out = subprocess.run([sys.executable, MODULE, "--check", p, "--quiet"],
                         capture_output=True, text=True, env=e)
    assert out.returncode == 0
    assert out.stdout.strip() == ""


# --------------------------------------------------------------------------
# the regression this whole module exists to prevent
# --------------------------------------------------------------------------

def test_the_2026_10_02_configuration_would_now_abort():
    """The exact paths the lost Paper B run wrote to.

    docs/PAPERB_PILOT_REPORT.md §8 listed these as "on the overlay, NOT
    durable" and the run proceeded anyway. It would now refuse to start.
    """
    lost = [
        "/tmp/specmem/paperb/pilot25_title_exact",
        "/tmp/specmem/paperb/pilot25b_bm25",
        "/tmp/specmem/paperb/scale100_title_exact",
        "/tmp/specmem/paperb/probe_noprefix.json",
        "/tmp/specmem/invariant",
        "/tmp/specmem/logs",
    ]
    for path in lost:
        with pytest.raises(durability.EphemeralOutputError):
            durability.require_durable_outputs(out=path)
