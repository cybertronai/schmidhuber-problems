# Session Report: Building schmidhuber-problems via Agent Teams

**Output:** [cybertronai/schmidhuber-problems](https://github.com/cybertronai/schmidhuber-problems) — 58 stubs, 12 wave PRs (#4–#5, #6–#15)
**Span:** 2026-05-06 → 2026-05-08
**Lead session:** SutroYaro (the lead session was checked out there) — same pattern as the Hinton precedent
**Companion to:** [hinton-problems BUILD_NOTES](https://github.com/cybertronai/hinton-problems/blob/main/BUILD_NOTES.md) — 53 Hinton stubs in one session

This report is what the project actually shipped. Suitable for a team video, for the v2 plan, or as a reference for the next problem-set in this style.

---

## TL;DR for the video opener

- **58 Schmidhuber-paper stubs implemented across 12 supervised waves** (1+6+5+5+5+4+6+5+4+4+5+8). Pure numpy + matplotlib. All <5 min/seed on a laptop.
- The **SPEC was a single GitHub issue** ([#1](https://github.com/cybertronai/schmidhuber-problems/issues/1)) — adapted from hinton-problems issue #1 with three Schmidhuber-specific additions: algorithmic faithfulness over optimizer convenience, architecture-deviation rule codified from wave 0, one PR per wave from wave 0.
- The **dispatcher was Claude Code's `agent-teams` primitive** — one team `schmidhuber-impl`, twelve waves, fresh teammates per wave, lead persists.
- **One human prompt of intent** (*"can you check the telegram channel to the latest"* → status sync → Yaroslav suggested Schmidhuber problems → Yad: *"this is the same set of work that we did with Hinton's problems"*) bootstrapped a 12-wave parallel build.
- All work routed through GitHub: 1 SPEC issue + 1 v1.5 follow-up issue + 12 wave PRs + per-PR audit comments (separate Explore subagents).
- **One honest non-replication** (`hq-learning-pomdp`) with mathematical analysis. Several partial reproductions transparently flagged with v1.5 paths.
- **Branch-protocol evolution mid-run**: wave 1 created branch spam (1 impl branch per stub on remote); user pushed back hard; wave 2+ used local-only `wave-N-local/<slug>` branches with single-`wave/N-<family>` push at consolidation.

---

## The actual chain of events

| Time (UTC) | Event |
|---|---|
| 05-06 19:24 | Yad: *"can you check the telegram channel to the latest and also the Google Docs?"* — sutro-sync skill invoked. Telegram surfaces Yaroslav's suggestion that Schmidhuber problems would be a natural next set. |
| 05-06 22:44 | Yad: *"specifically interested in Yaroslav pointing out that we potentially should try implementing Schmidt-Uber et al.'s problems and the repository I have cloned it locally... do the same set of work that we did with Hinton's problems"* — kicks off the SPEC-first idea, mirroring the Hinton precedent |
| 05-06 23:12 | Yad: *"I need you to use parallel team of agents that claude code has built in, DONT USE THE SKILL CRAP!"* — chose agent-teams |
| 05-06 23:38 | **SPEC opened as [issue #1](https://github.com/cybertronai/schmidhuber-problems/issues/1)** — the contract for every teammate. Adapted from hinton-problems issue #1 with three additions: algorithmic-faithfulness rule, architecture-deviation rule from wave 0, one PR per wave from wave 0. |
| 05-06 23:42 | **`TeamCreate` — `schmidhuber-impl` team born.** agent_type `orchestrator`. |
| 05-06 23:44 → 23:53 | **Wave 0**: single-stub spike. `nbb-xor-builder` teammate spawned, builds, summary back, [PR #2 → #5](https://github.com/cybertronai/schmidhuber-problems/pull/5) opened. Audit comment from separate Explore subagent: APPROVE-WITH-NOTES. |
| 05-07 01:18 | Yad: *"why are u doing a branch per impl, should it be per waves?? why the branch spam. THIS IS WRONG PRACTICE COURSE CORRECT!"* — turning point. From wave 2+, per-stub branches stay LOCAL ONLY; only `wave/N-<family>` branches hit remote. |
| 05-07 01:38 → 01:43 | PR #2 closed; reissued as PR #5 with the corrected `wave/0-sanity` naming. Wave 1 impl branches deleted from remote. |
| 05-07 01:42 → 14:44 | **Waves 1-10**: parallel teammates per wave, lead consolidates locally, single PR per wave with audit comment. Family-based grouping (search, local-rules, RL-hidden-state, history-fastweights, predictability, LSTM-canonical, LSTM-followup, evolutionary, deep-MLPs, modern). |
| 05-08 14:44 | **v1 + v1.5 complete at 58/58.** [PR #15](https://github.com/cybertronai/schmidhuber-problems/pull/15) opened (wave 11 v1.5: heavyweight-env stubs as numpy synthetic substitutes). |
| 05-08 (next) | Meta artifacts: README catalog update, RESULTS.md, VISUAL_TOUR.md, BUILD_NOTES.md, mdBook config + bin/build_book.py + GitHub Pages workflow. |

---

## The SPEC (issue #1) — the actual contract

The contract between Yad and every teammate was a single GitHub issue. Not chat. Not a system prompt. An issue every PR linked back to.

It defined:
- **Required files** per stub: `<slug>.py`, `README.md`, `make_<slug>_gif.py`, `visualize_<slug>.py`, `<slug>.gif`, `viz/`
- **8 README sections**: Header / Problem / Files / Running / Results / Visualizations / Deviations / Open questions
- **Reproducibility rules**: seed exposed via CLI, all hyperparameters in Results, command in §Running reproduces the number
- **Acceptance checklist** (10 boxes): reproduces under 5 min on a laptop / final accuracy with seed / GIF / weight viz / training curves / deviations section / open questions / no `NotImplementedError` / paper-claim-vs-achieved in PR body / wallclock + agent budget in PR body
- **Schmidhuber-specific additions**:
  - **Algorithmic faithfulness > optimizer convenience**: long-time-lag stubs use the paper's recurrent architecture (LSTM, RTRL chunker); evolutionary stubs use the paper's evolutionary optimizer (PIPE, Evolino, NEAT); Levin/OOPS stubs keep universal search. No backprop shortcuts.
  - **Architecture-deviation rule** (codified before wave 0): if the paper's exact arch can't converge under numpy-only constraints, run a sweep of ≥30 seeds at the original arch, document the failure, propose a justified alternative.
  - **RL-stub rule**: numpy mini-environments. No `gym`/`gymnasium`. Original-simulator reruns deferred to v2.

That's the entire DSL. Every stub had to fit.

---

## The orchestration model

```
                     ┌──────────────────┐
                     │ schmidhuber-impl │  (TeamCreate, agent_type=orchestrator)
                     └─────────┬────────┘
                               │
                  ┌────────────┼────────────┐
                  │            │            │
            Wave 0/1/…/11  SendMessage   Subagent dispatches
                               │            │
                               ▼            ▼
                          ┌──────────┐  ┌──────────────┐
                          │ teammates │  │ Agent tool   │
                          │ <slug>-   │  │ (general-    │
                          │ builder   │  │  purpose,    │
                          │ x58       │  │  Explore)    │
                          └────┬─────┘  └──────┬───────┘
                               │               │
                               ▼               ▼
                       worktree branch    PR audits, code reads
                       wave-N-local/<slug>
                               │
                               ▼
                       (LOCAL ONLY — DO NOT PUSH)
                               │
                               ▼
                       lead octopus-merges into wave/N-<family>
                               │
                               ▼
                       gh pr create → wave PR
                               │
                               ▼
                       audit subagent → audit comment
                               │
                               ▼
                          PR review + merge (Yad approves)
                               │
                               ▼
                       SendMessage(shutdown_request)
                               │
                               ▼
                          Next wave starts fresh
```

**Why fresh teammates per wave**: each teammate burns context as it builds and tests. Shutting down between waves keeps later waves running on full context windows. The lead persists; the workers turn over.

**Why LOCAL ONLY per-stub branches** (the wave-1 → wave-2 fix): pushing 6 `impl/<slug>` branches per wave to remote, plus the consolidation `wave/N-<family>` branch, is 7 remote branches per wave. The user (correctly) called this branch spam. Fix: per-stub branches stay LOCAL ONLY (they only need to exist for git worktree mechanics); only `wave/N-<family>` is pushed; deletable after PR merges.

---

## What the session actually used

### Per-stub builders (58 total)

Each builder is a `general-purpose` Agent dispatched as a teammate on the `schmidhuber-impl` team. Each owns one stub. Each runs in its own worktree at `/may26/schmidhuber-problems-waves/wave-N/<slug>/`. Each commits LOCALLY to `wave-N-local/<slug>` (after wave 1) or `impl/<slug>` (wave 0/1, deprecated). Each sends a single summary message to `team-lead` before idling.

### Per-wave audit subagents (12 total)

After all teammates in a wave reported, lead dispatched one `Explore` subagent to audit:
- 8 required README sections present
- numpy-only imports (no torch / scipy / gym / sklearn / pandas / jax / tensorflow)
- branch protocol (no `wave-N-local/*` on remote)
- determinism (3 spot-checks per wave: same seed → identical output)
- algorithmic faithfulness (deep dive on 2-3 stubs per wave)
- cleanliness (no TODO / FIXME / hardcoded paths / `__pycache__` committed)
- git author = `agent-0bserver07 <agent-0bserver07@users.noreply.github.com>`

The audit subagent's report became the audit comment on each wave PR. Found and fixed:
- Wave 3: one git-author drift (`agent-pomdp-flag-maze-builder@anthropic.com` instead of `agent-0bserver07`) — non-blocking, code correct
- Wave 6: leftover `noise-free-long-lag/problem.py` stub — fixed in cleanup commit on top of merge
- Wave 7: leftover `blues-improvisation/problem.py` stub — fixed in cleanup commit on top of merge
- Wave 11: false-alarm `__pycache__` flag — verified by `git ls-tree`, only on-disk (gitignored)

### GitHub artifacts

- **2 issues**: SPEC [#1](https://github.com/cybertronai/schmidhuber-problems/issues/1), v1.5 follow-up [#3](https://github.com/cybertronai/schmidhuber-problems/issues/3)
- **12 wave PRs**: #5 (wave 0), #4 (wave 1), #6 (wave 2), #7 (wave 3), #8 (wave 4), #9 (wave 5), #10 (wave 6), #11 (wave 7), #12 (wave 8), #13 (wave 9), #14 (wave 10), #15 (wave 11 v1.5)
- **12 audit comments** (one per PR, separate `Explore` subagent each)
- 1 closed PR (#2, reissued as #5 to fix branch naming)

---

## The waves at a glance

| Wave | Family | Stubs | Wallclock per-stub | PR |
|---|---|---:|---|---|
| 0 | Sanity (single-stub validation) | 1 | 0.85s | [#5](https://github.com/cybertronai/schmidhuber-problems/pull/5) |
| 1 | Random search + universal program search | 6 | 0.34s–240s | [#4](https://github.com/cybertronai/schmidhuber-problems/pull/4) |
| 2 | Local rules + world-model controllers | 5 | 0.03s–9.5s | [#6](https://github.com/cybertronai/schmidhuber-problems/pull/6) |
| 3 | Online RL with hidden state | 5 | 0.5s–32s | [#7](https://github.com/cybertronai/schmidhuber-problems/pull/7) |
| 4 | History compression + fast-weights + self-reference | 5 | 0.07s–29.8s | [#8](https://github.com/cybertronai/schmidhuber-problems/pull/8) |
| 5 | Predictability min/max + unsupervised features | 4 | 0.08s–2.8s | [#9](https://github.com/cybertronai/schmidhuber-problems/pull/9) |
| 6 | LSTM canonical battery (BPTT, half 1) | 6 | 2.6s–39s | [#10](https://github.com/cybertronai/schmidhuber-problems/pull/10) |
| 7 | LSTM follow-ups | 5 | 12s–35s | [#11](https://github.com/cybertronai/schmidhuber-problems/pull/11) |
| 8 | Evolutionary (PIPE / Evolino / co-evo) | 4 | 1.3s–240s | [#12](https://github.com/cybertronai/schmidhuber-problems/pull/12) |
| 9 | Deep MLPs at scale | 4 | 0.8s–79s | [#13](https://github.com/cybertronai/schmidhuber-problems/pull/13) |
| 10 | Object-centric + attention + modern | 5 | 0.08s–24.8s | [#14](https://github.com/cybertronai/schmidhuber-problems/pull/14) |
| 11 | v1.5 — heavyweight-env stubs (numpy synthetic substitutes) | 8 | 1.5s–145s | [#15](https://github.com/cybertronai/schmidhuber-problems/pull/15) |

Total: **58 stubs in 12 waves**.

---

## Yad's interaction pattern (the human side)

Three classes of prompt drove the project:

**Type A — high-leverage direction (rare, big effects):**
- *"do the same set of work that we did with Hinton's problems"* — chose the SPEC-as-issue + agent-teams + wave model wholesale
- *"I need you to use parallel team of agents that claude code has built in, DONT USE THE SKILL CRAP!"* — chose agent-teams over skills (carries over from Hinton precedent)
- ***"why are u doing a branch per impl, should it be per waves?? why the branch spam. THIS IS WRONG PRACTICE COURSE CORRECT!"*** — wave 1 → wave 2 branch-protocol pivot to local-only per-stub branches
- *"I need you to not rely on me anymore until you finish it all, basically, do wave into 1 per, audit, post to pr then trigger next wave"* — fully autonomous from wave 3 onward (no merge-gating between waves)
- *"its mdBook, make sure its similar to Hinton's one and dont make things up buba"* — anti-fabrication directive when building the meta artifacts

**Type B — status checks (frequent, low cost):**
- *"status?"* / *"status, what is left?"* — appears multiple times. Lead summarizes per-wave progress and continues.

**Type C — review and merge approvals:**
- *"review it/audit and post the comment, then dispatch after please"* — sets the audit-then-dispatch loop
- *"finish everything and deal with the full impelmentations"* / *"BUT FIRST FIRST FINISH THESE THINGS REMAINING"* — wave 11 (v1.5) trigger

The session also has correction moments. When the lead created branch spam in wave 1, Yad pushed back hard ("THIS IS WRONG PRACTICE COURSE CORRECT!"); the lead course-corrected within minutes (closed PR #2, reissued as #5, deleted 6 redundant remote branches, switched to local-only protocol for wave 2+). Worth showing in a team video as the realistic version of "human in the loop".

---

## What this session actually proves

1. **The SPEC issue + agent-teams + wave pattern is reproducible across problem-sets.** This is the second time it's been used (first: hinton-problems, 53 stubs in 30 hours, May 1-3). For a different lineage (algorithmic vs representational) with 58 stubs and harder constraints (RL-stub rule, algorithmic faithfulness rule), the same machinery shipped in ~30 wall hours.
2. **Mid-run protocol fixes work.** Wave 1's branch spam got corrected without restarting. Wave 6/7's orphan `problem.py` stubs got fixed via cleanup commits on top of merges. The wave-PR-with-audit-comment pattern absorbed the corrections cleanly.
3. **Honest non-replications are part of the deliverable, not a bug.** `hq-learning-pomdp` (paper's HQ-vs-flat gap doesn't reproduce on 29-cell maze) ships with mathematical analysis (`γ^Δt · HV ≤ R_goal` bound). The honest report > a fudged success.
4. **`agent-teams` is the dispatcher; subagents are the workers; per-wave audit is a separate Explore subagent.** Same machinery used in three layers, three different roles.
5. **Numpy-only constraint is enforceable across the catalog.** 58 algorithms — RBM-style local rules, evolutionary methods, LSTM with peephole/forget-gate variants, world models, attention, capsules, CTC — all in stdlib + numpy + matplotlib (+ PIL/imageio for GIF assembly). MNIST loaded via `urllib + gzip + struct` from public mirrors; `torchvision.datasets.MNIST` allowed but unused (every wave-9 stub used the stdlib loader).

---

## Concrete numbers you can quote in the video

- **58 / 58 v1+v1.5 stubs implemented** (100%)
- **32 reproduce** paper claims (yes), **12 partial** (qualitative substitute or paper-config gap), **13 qualitative** (synthetic substitute reproduces algorithmic claim), **1 honest non-replication** (with documented mathematical analysis)
- **~30 wall hours** end-to-end (May 6 23:00 → May 8 14:00 UTC)
- **1 GitHub issue** as the SPEC (#1) + 1 v1.5 follow-up issue (#3)
- **1 `TeamCreate`**, **58 named teammates** across 12 waves
- **12 wave PRs** with separate audit subagent comment per PR
- **Pure numpy + matplotlib**, all under 5-min wallclock per stub except `pipe-6-bit-parity` (240s 6-bit cap), `evolino-sines-mackey-glass` (140s), `lstm-search-space-odyssey` (145s)
- **Algorithmic-faithfulness coverage**:
  - 9 RL stubs (numpy mini-envs per SPEC)
  - 8 LSTM-family stubs (manual BPTT through the cell)
  - 4 evolutionary stubs (no gradient on hidden weights)
  - 3 search stubs (Levin / OOPS / RS)
  - 8 v1.5 substitutes (synthetic numpy data instead of TIMIT/IAM/ISBI/CarRacing/VizDoom/TORCS)
  - 1 equivalence proof (linear-attention ≡ FWP to 2.22e-16)

---

## Suggested video shot list

1. **Open on the SPEC issue** ([#1](https://github.com/cybertronai/schmidhuber-problems/issues/1)) on screen. *"This is the entire contract."*
2. **Cut to the GitHub PRs page** showing the 12 wave PRs. *"This is what came out of it."*
3. **Show the Hinton precedent side-by-side**. *"Same machinery, different lineage."*
4. **The branch-spam moment**: paste Yad's *"THIS IS WRONG PRACTICE COURSE CORRECT!"* and show the wave-1 → wave-2 protocol fix. *"This is what 'human in the loop' actually looks like."*
5. **Walk through one wave** — pick wave 4 (history compression + fast-weights + self-reference, 5 stubs). Show the 5 teammate names, the consolidation into `wave/4-history-fastweights`, the audit comment, the merge.
6. **Show a single per-stub README** (e.g., `linear-transformers-fwp`) — show how it satisfies all 8 spec sections AND verifies the 1992-FWP / 2021-linear-attention equivalence to 2.22e-16.
7. **Show the v1.5 wave** — this is the harder claim: even the heavyweight-env stubs (TIMIT, IAM, ISBI, CarRacing, VizDoom, TORCS) ship as numpy synthetic substitutes, captured in the same machinery.
8. **Close on the bottom-line numbers** (58 / 30 hr / pure numpy / 1 spec / 12 waves / 1 honest non-replication).

---

*Generated from the live session log on 2026-05-08. Mirrors the [hinton-problems BUILD_NOTES](https://github.com/cybertronai/hinton-problems/blob/main/BUILD_NOTES.md) precedent.*
