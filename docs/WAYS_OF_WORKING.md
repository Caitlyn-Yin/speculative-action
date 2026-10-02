# WAYS_OF_WORKING

Standing rules. These bind every session, including the ones that think they are in a hurry.

## 1. Done means pushed

**A step is complete only when its commits AND its tag are on `origin`.**

- Local commits are work-in-progress, regardless of how finished they look.
- **Every report ends with the pushed commit hash.** If nothing was pushed, the report says so in
  those words and names the blocker — it does not present local hashes as if they were durable.
- Bundles are **disaster-recovery only** and **must exist off-node**. A bundle on the same host as
  the repo — including on JuiceFS home — is not a backup; it shares the failure domain with the
  thing it is backing up.

Origin of this rule: the 2026-09 loss. Steps 1–2.5 and Phases A/B were completed on the GH200 and
never pushed. When that host became unreachable, the work was gone — code, contracts and
adjudications alike (`LOST_WORK_MANIFEST.md`). The commits looked done on the machine where they
lived, which is exactly the trap.

## 2. Corollaries

- Push **before** starting the next step, not at the end of the session.
- A tag without its branch on `origin` is not durable either; push both, then verify with
  `git ls-remote origin` and paste the output.
- If auth blocks a push: do not wait and do not proceed as if it succeeded. Print the exact error,
  get the bundle off the node, and report the blocker at the top of the response.
- Artifacts that cannot be committed (weights, caches, trajectories) must be **reproducible from a
  committed script**, and the script must record what would be needed to rebuild them.

## 2a. Durability rules

**Every output goes on persistent storage, and every task ends with `scripts/checkpoint.sh`.**

Rule 1 covers *code*. It did not cover *data*, and data is how the project lost work twice:

| When | What was lost | Mechanism |
|---|---|---|
| 2026-09 | Steps 1–2.5, Phases A/B, the judge contract, the 29-window ladder | committed on the GH200, never pushed; host became unreachable |
| 2026-10-02 | the **entire** Paper B corpus — both pilot runs, both backends, and the 100-question scale-up that was still running | all of it under `/tmp/specmem/paperb`; pod recycled |

The second one is the expensive lesson: the pilot report's §8 *correctly listed* those artifacts as
"on the overlay, NOT durable" and the run proceeded anyway. Knowing a path is ephemeral does not
save the data. Only the mechanics below do.

### The two roots

`scripts/env.sh` is the single source of truth, and it defines exactly two:

| Variable | Mount | Survives recycle | Holds |
|---|---|---|---|
| `$SPEC_BASE` = `$HOME/specmem-data` | JuiceFS | **yes** | **every output**: `runs/`, windows, replay results, criterion scores, judge + embedding caches, logs |
| `$SPEC_SCRATCH` = `/tmp/specmem` | overlay | no | **read-only inputs and rebuildable tooling only**: conda envs, model weights, XDG dirs, raw corpus download |

Read-only inputs **may** be staged on scratch for speed — that is what it is for, and
`setup_envs.sh` / `download_ladder.sh` put them back after a recycle. An **output** on scratch is
simply a bug.

`$SPEC_BASE` is a 64 GB JuiceFS quota that is already ~87 % full (the 7.6 GiB frozen corpus is most
of it). Outputs must stay jsonl/json-sized; weights must never go there.

### The guard

`hotpotqa/src/durability.py` is imported by every entry script and called **before any compute**.
It aborts the run if an output path lies under `/tmp`, `/dev/shm`, `/var/tmp`, `/run`, or on a mount
`findmnt -T` reports as `tmpfs`/`overlay`. Shell entry scripts call the same module through
`spec_require_durable` (defined in `env.sh`, run with the *system* `python3` so the guard works on a
freshly recycled pod where `$PIPELINE_PY` does not exist yet).

It fails closed and it fails early: a misconfigured `--out` costs zero GPU time instead of a
500-second audit. Do not route around it — if a path it rejects is genuinely fine, fix the path.

### The checkpoint

**Every task ends by running `bash scripts/checkpoint.sh`.** It copies every `.jsonl`/`.json`/`.md`
file under `$SPEC_RUNS_DIR` smaller than 50 MB into `results/<run_id>/`, commits, and pushes over
the deploy key. Anything at or above the limit is **listed with its path and size**, never silently
dropped.

Persistent storage and git are not redundant — they fail differently. JuiceFS stops the recycle
(2026-10-02). Only `git push` stops the node-goes-away (2026-09). `$SPEC_BASE` is one quota on one
node; a run that is only there is one `efabric` incident from being the third entry in that table.

Corollaries:

- Put run artifacts under `$SPEC_RUNS_DIR/<run_id>/`. An output elsewhere under `$SPEC_BASE`
  survives a recycle but is **invisible to the sweep**, so it never reaches git.
- Write machine-readable artifacts as `.json`/`.jsonl`, not `.txt` — the sweep is extension-based,
  and a `.txt` config record is evidence that silently stays behind.
- `--no-push` leaves the work local only, which §1 does not count as done. If the push fails, report
  the blocker; do not call the task finished.

## 3. How to push from EFabric

**Push over SSH with the repo deploy key — `gh`/HTTPS can never work on this node.**

```bash
git remote set-url origin git@github.com-specmem:Caitlyn-Yin/speculative-action.git
git push origin phase2-fixed-scaleup && git push origin --tags
```

The alias is in `~/.ssh/config` (`Host github.com-specmem` → `HostName github.com`, `User git`,
`IdentityFile ~/.ssh/id_specmem_deploy`, `IdentitiesOnly yes`). The key is a **repository deploy
key with write access**, not an account key, so its blast radius is this one repo. If a future pod
blocks port 22, switch that entry to `HostName ssh.github.com` / `Port 443` — both were reachable
on 2026-09-28.

**Why not `gh`:** `$HOME` is the root of a JuiceFS mount, and JuiceFS materialises `.accesslog`,
`.stats` and `.config` as root-owned *virtual files* at its mount root. So `~/.config` is a
0400 root-owned **regular file** that no one can delete or convert — every mount, forever. Anything
resolving an XDG path beneath it dies (`gh`: `open ~/.config/gh/config.yml: not a directory`;
vLLM: `NotADirectoryError`). It is not a cron artifact and there is nothing to repair — its mtime
just tracks pod start. `GH_CONFIG_DIR=$HOME/.gh-config` does make `gh` start, but it puts an
account-scoped token on a shared node; the deploy key is preferred. `env.sh` redirects
`XDG_CONFIG_HOME`/`XDG_CACHE_HOME` for the same underlying reason.

`~/.ssh/config` is partly managed by EFabric (`# EFabric worker ssh aliases begin/end`). The
github block sits outside that region and survived the 2026-09-25 pod restart, but **verify it is
still present after any restart** before concluding that auth broke.

**After any pod restart, run `ssh -T git@github.com-specmem` before assuming anything about git.**
Expect `Hi Caitlyn-Yin/speculative-action!` (deploy key). A bare `Hi Caitlyn-Yin!` means the key is
registered account-wide rather than as a repo deploy key — it pushes fine, but with a much larger
blast radius than intended.
