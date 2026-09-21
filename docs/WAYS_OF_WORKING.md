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
