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
