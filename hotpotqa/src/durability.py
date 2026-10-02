"""Refuse to write run outputs to storage that a pod recycle will wipe.

Every output loss on this project came from writing to ``/tmp``:

* 2026-09 (GH200): Steps 1-2.5 and Phases A/B -- code, contracts and
  adjudications. See ``docs/LOST_WORK_MANIFEST.md``.
* 2026-10-02: the entire Paper B pilot corpus (both runs, both backends) and the
  100-question scale-up that was still running, all under
  ``/tmp/specmem/paperb``. Nothing was recoverable; only the numbers transcribed
  into ``docs/PAPERB_PILOT_REPORT.md`` survive.

Both times the artifacts were *reproducible in principle* and gone in practice.
This module is the mechanical guard that stops the third time: every entry
script declares the paths it is about to write, and a path on an ephemeral mount
aborts the run **before** any compute is spent rather than after.

Policy (``scripts/env.sh`` is the single source of truth for the paths):

* ``$SPEC_BASE``    persistent (JuiceFS) -- every output.
* ``$SPEC_SCRATCH`` ephemeral (overlay)  -- read-only inputs and rebuildable
  tooling only: conda envs, model weights, XDG dirs, raw downloads.

A mount counts as ephemeral if ``findmnt -T`` reports a filesystem type in
``EPHEMERAL_FSTYPES`` (``tmpfs``, ``overlay``, ...), or if the path lies under a
known-ephemeral prefix such as ``/tmp`` or ``/dev/shm``. The prefix check is not
redundant: it still fires when ``findmnt`` is unavailable or when ``/tmp`` is a
plain directory on a durable root.

Usage from an entry script, before doing any work::

    from src import durability
    durability.require_durable_outputs(out_dir=args.out, log=args.log_path)

and from a shell entry script::

    python3 "$REPO/hotpotqa/src/durability.py" --check "$OUT_ROOT" "$LOG_DIR" || exit 1

Deliberately stdlib-only and importable without the pipeline env, so the guard
is available on a freshly recycled pod where nothing has been installed yet.
"""

import argparse
import os
import subprocess
import sys

#: Filesystem types that do not survive a pod recycle on this platform.
#: ``overlay`` is the container root; ``tmpfs``/``ramfs`` are memory-backed.
EPHEMERAL_FSTYPES = frozenset({
    "tmpfs", "ramfs", "devtmpfs", "overlay", "overlayfs", "squashfs",
})

#: Path prefixes that are ephemeral by convention even when the mount lookup
#: says otherwise (e.g. ``/tmp`` as a directory on a durable root).
EPHEMERAL_PREFIXES = ("/tmp", "/dev/shm", "/var/tmp", "/run")

#: Filesystem types known to persist across a recycle on this node.
DURABLE_FSTYPES = frozenset({
    "fuse.juicefs", "juicefs", "nfs", "nfs4", "ext4", "xfs", "btrfs", "lustre",
})


#: Escape hatch, for **tests only**.
#:
#: The offline suites build throwaway fixtures (a 6-page mini KILT corpus, a
#: scripted corpus) in pytest's ``tmp_path``, which is under ``/tmp`` by design.
#: Those are not run outputs and making them durable would write fixture data to
#: a 64 GB quota on every test run. ``tests/conftest.py`` sets this.
#:
#: It is deliberately not a general bypass: it announces itself on every call,
#: so an ephemeral path in a real run is visible in the log rather than silent.
#: If you are reaching for this outside a test, the path is the thing to fix.
ALLOW_ENV_VAR = "SPEC_ALLOW_EPHEMERAL_OUTPUTS"


def _bypass_active():
    return os.environ.get(ALLOW_ENV_VAR) == "1"


class EphemeralOutputError(RuntimeError):
    """An output path resolves to storage a pod recycle will wipe."""


def _nearest_existing(path):
    """Deepest existing ancestor of ``path`` (the mount is a property of it)."""
    p = os.path.abspath(path)
    while not os.path.exists(p):
        parent = os.path.dirname(p)
        if parent == p:
            return p
        p = parent
    return p


def _findmnt(path):
    """``(fstype, mountpoint)`` for ``path`` via findmnt, or ``(None, None)``."""
    try:
        out = subprocess.run(
            ["findmnt", "-T", path, "-n", "-o", "FSTYPE,TARGET"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None, None
    if out.returncode != 0:
        return None, None
    fields = out.stdout.split()
    if len(fields) < 2:
        return None, None
    return fields[0], fields[1]


def _proc_mounts(path):
    """Fallback: longest-prefix match against /proc/mounts."""
    best = (None, None, -1)
    try:
        with open("/proc/mounts", encoding="utf-8") as fh:
            lines = fh.readlines()
    except OSError:
        return None, None
    for line in lines:
        parts = line.split()
        if len(parts) < 3:
            continue
        target, fstype = parts[1], parts[2]
        if path == target or path.startswith(target.rstrip("/") + "/"):
            if len(target) > best[2]:
                best = (fstype, target, len(target))
    return best[0], best[1]


def classify(path):
    """Describe where ``path`` would actually be written.

    Returns a dict with ``path``, ``probed`` (the ancestor whose mount was
    inspected), ``fstype``, ``mountpoint``, ``durable`` and ``reason``.
    Non-existent paths are classified by their nearest existing ancestor, so
    this works before a run has created its output directory.
    """
    abspath = os.path.abspath(path)
    probed = _nearest_existing(abspath)

    fstype, mountpoint = _findmnt(probed)
    if fstype is None:
        fstype, mountpoint = _proc_mounts(probed)

    prefix_hit = next(
        (pre for pre in EPHEMERAL_PREFIXES
         if abspath == pre or abspath.startswith(pre.rstrip("/") + "/")),
        None,
    )

    if prefix_hit is not None:
        return dict(path=abspath, probed=probed, fstype=fstype,
                    mountpoint=mountpoint, durable=False,
                    reason="under the ephemeral prefix %s" % prefix_hit)
    if fstype is not None and fstype in EPHEMERAL_FSTYPES:
        return dict(path=abspath, probed=probed, fstype=fstype,
                    mountpoint=mountpoint, durable=False,
                    reason="mount %s is %s, which a pod recycle wipes"
                           % (mountpoint, fstype))
    if fstype is None:
        return dict(path=abspath, probed=probed, fstype=None,
                    mountpoint=None, durable=True,
                    reason="mount type could not be determined; assumed durable")
    return dict(path=abspath, probed=probed, fstype=fstype,
                mountpoint=mountpoint, durable=True,
                reason="mount %s is %s" % (mountpoint, fstype))


def is_durable(path):
    return classify(path)["durable"]


def _advice():
    spec_base = os.environ.get("SPEC_BASE")
    if spec_base:
        return ("Write it under $SPEC_BASE (%s) instead -- for run artifacts, "
                "under $SPEC_RUNS_DIR (%s) so scripts/checkpoint.sh picks it up."
                % (spec_base, os.environ.get("SPEC_RUNS_DIR",
                                             os.path.join(spec_base, "runs"))))
    return ("SPEC_BASE is not set: `source scripts/env.sh` first, then write "
            "outputs under $SPEC_RUNS_DIR.")


def require_durable_outputs(**paths):
    """Abort unless every named output path is on persistent storage.

    Keyword names are used in the error message, so pass something meaningful::

        durability.require_durable_outputs(out_dir=args.out, cache=args.cache_dir)

    Returns the dict of absolute paths on success. Raises
    :class:`EphemeralOutputError` listing every offending path -- all of them,
    not just the first, so one run fixes the whole script.
    """
    resolved, bad = {}, []
    for name, path in paths.items():
        if path is None:
            continue
        info = classify(path)
        resolved[name] = info["path"]
        if not info["durable"]:
            bad.append((name, info))

    if bad:
        lines = ["refusing to run: %d output path(s) resolve to ephemeral "
                 "storage that a pod recycle will wipe." % len(bad)]
        for name, info in bad:
            lines.append("  %s = %s" % (name, info["path"]))
            lines.append("      %s" % info["reason"])

        if _bypass_active():
            print("  durability: WARNING: %s=1 -- allowing %d ephemeral output "
                  "path(s) that would otherwise abort this run:"
                  % (ALLOW_ENV_VAR, len(bad)), file=sys.stderr)
            for name, info in bad:
                print("    %s = %s (%s)" % (name, info["path"], info["reason"]),
                      file=sys.stderr)
            print("  durability: anything written there will NOT survive a pod "
                  "recycle. This is for test fixtures only.", file=sys.stderr)
            return resolved

        lines.append("")
        lines.append(_advice())
        lines.append("Rationale and history: docs/WAYS_OF_WORKING.md "
                     "'Durability rules'.")
        raise EphemeralOutputError("\n".join(lines))
    return resolved


def warn_if_outside_spec_base(**paths):
    """Print a warning for outputs that are durable but outside ``$SPEC_BASE``.

    Not fatal: the repo working tree is durable, and docs legitimately live
    there. But an output outside ``$SPEC_RUNS_DIR`` is invisible to
    ``scripts/checkpoint.sh``, so it never reaches git.
    """
    spec_base = os.environ.get("SPEC_BASE")
    if not spec_base:
        return
    root = os.path.abspath(spec_base).rstrip("/") + "/"
    for name, path in paths.items():
        if path is None:
            continue
        abspath = os.path.abspath(path)
        if not abspath.startswith(root):
            print("  durability: note: %s = %s is outside $SPEC_BASE, so "
                  "scripts/checkpoint.sh will not see it" % (name, abspath),
                  file=sys.stderr)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Check that output paths are on persistent storage.")
    ap.add_argument("--check", nargs="+", metavar="PATH", required=True,
                    help="output paths this run intends to write")
    ap.add_argument("--quiet", action="store_true",
                    help="print nothing when every path is durable")
    args = ap.parse_args(argv)

    try:
        require_durable_outputs(**{"path%d" % i: p
                                   for i, p in enumerate(args.check, 1)})
    except EphemeralOutputError as exc:
        print("durability: %s" % exc, file=sys.stderr)
        return 1
    if not args.quiet:
        for p in args.check:
            info = classify(p)
            print("durability: ok  %s  (%s)" % (info["path"], info["reason"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
