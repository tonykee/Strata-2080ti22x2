# Prompt lookup without DSpark (DeepSeek-V4.1-Flash)

The DeepSeek-V4.1-Flash engine can draft from the prompt lookup alone, with no DSpark drafter in the pack. It
needs no extra model and no extra graphics memory. This page records the engine change, how to turn it on, and
what it measured on two RTX 2080 Ti.

**Status:** on by default in the launchers. The CLI builds the lookup when `--suffix-draft` is above 0
(default 3), and `start_ds4_gsq_server.sh` / `start_ds4_gsq_2080ti.sh` export `DS4_LOOKUP_ALONE=1`, which
`serve()` needs. `--suffix-draft 0` turns the lookup off; `DS4_LOOKUP_ALONE=0` turns off the server's switch.
In the greedy runs below the 120 output token ids were identical with the lookup off and on, so the drafts do
not change the answer.

## What it does

The prompt lookup (the suffix drafter) searches the conversation for the text being generated and proposes the
tokens that followed it last time. It is the second drafter beside DSpark: it has no weights and runs on the
host. Its proposals are checked by the same verify pass that checks DSpark's, and only the tokens the model
agrees with are kept.

## Why it did not run before

`serve()` built the lookup only when DSpark was on (`if (spec && suffix_draft > 0)`), and `round()` entered its
drafting block only for `spec && m.draft_block() > 0`. A pack without the drafter therefore never took a lookup
round, even with `--suffix-draft` set.

Forcing the lookup on found a second problem. `Model::verify()` uses three device buffers, `d_ints`, `d_confv`
and `d_specq`, and they were allocated only when the pack carried the drafter (`g.n_mtp > 0`). With no drafter
they were null, and the first lookup window died with:

```text
maya-ds4: cudaMemcpyAsync(h_idx, d_ints + 16, ...): an illegal memory access was encountered
```

The same prompt with `--suffix-draft 0` ran normally, so the fault was the lookup window, not the model or the
prompt.

Fixing that exposed a second MTP-only allocation. The verify window keeps its rollback rows in `ring_bak`, and the
compressor's state in `c_snap` / `c_rows`; those were allocated only when `g.n_mtp > 0` too. A short prompt never
wraps the KV ring and does not write the rollback rows, so the 120-token runs passed. A long prompt does: 32,645 and
52,232 tokens both died after two tokens with the same illegal memory access. The 47,221-token request that reported
the failure is this case.

## The change

- `round()` (src/dsv41/ds4_main.cpp) runs the drafting block when the lookup exists as well as when DSpark does.
  Without a drafter the MTP part is skipped and the existing lookup-alone window carries the round: the lookup
  proposes up to 4 tokens, verify checks them, and the accepted ones are committed with the next token the model
  writes.
- `serve()` builds the lookup for a pack without a drafter when `DS4_LOOKUP_ALONE=1`. The CLI builds it whenever
  `--suffix-draft N` is above 0, as it did before.
- `d_ints`, `d_confv` and `d_specq` are allocated on the head card whether or not the pack has a drafter
  (src/dsv41/ds4_model.cu). They are a few KB; the verify path needs them for every window.
- The verify window's rollback rows (`ring_bak`) and the compressor's snapshot (`c_snap`, `c_rows`) are allocated
  for every non-MTP layer, not only when the pack has a drafter. They are under a megabyte a card.

The four commits are `8625f5809`, `1ddf636df`, `b3554113f` and `e2f2dfcee` on the `ds4-cuda-prefill` branch (the
second was an intermediate guard that the third replaced; the fourth adds the rollback rows above).

## Turning it on

CLI:

```sh
./start_ds4_gsq_2080ti.sh --ids "..." --n 120          # --suffix-draft 3 is the default
DS4_SUFFIX_DRAFT=0 ./start_ds4_gsq_2080ti.sh ...       # the lookup off
```

Server:

```sh
./start_ds4_gsq_server.sh                              # DS4_LOOKUP_ALONE=1 by default
```

When the engine or `serve/server.py` is started without these scripts, add `DS4_LOOKUP_ALONE=1` to the
environment (or to the config's `env`) and pass `--suffix-draft 3` in the args. A config with
`--suffix-draft 0` turns the lookup itself off.

## Measured

Host: 2x RTX 2080 Ti 22 GB (220 W), Xeon, 256 GB RAM, TP2, 20 CPU threads. Pack: UD-IQ1_M (Unsloth, 1.75 bpw,
vision, no DSpark drafter), 4 experts, 32K context, greedy, 120 tokens. Prompt: "Here is a text. Repeat it back
to me verbatim, word for word, without adding anything:" followed by one paragraph, so the answer copies the
prompt.

| `--suffix-draft` | Decode, 120 tokens | Prompt lookup |
| --- | ---: | --- |
| 0 (off) | 7.41, 7.43, 7.52 tok/s | - |
| 3 (on) | 8.89, 9.19, 12.93 tok/s | 24-25 rounds, 91-92 of 92 drafts kept |

These are single loads and the cards were not in the same state each time, so the range is wide. The highest
on-run came right before the 7.52 off-run and is +72%; 9.19 against the 7.41 off-run is +24%. What is stable is
the acceptance: the lookup kept every draft it offered in these runs (91-92 of 92 across 24-25 rounds). All six
runs produced the same 120 token ids, so the speed was not bought with a different answer.

In an engine `--serve` run with `DS4_LOOKUP_ALONE=1` the lookup also fired: the last 120-token request
generated in 8.1 s (14.75 tok/s) and kept 77 of 77 drafts. That run reused the prompt (151 of 152 tokens came
from the prefix cache) and the cards were warmer, so the number is not comparable with the CLI table above; it
shows that the server path builds and uses the lookup.

### Long prompts

Same box, TP1 (layer split), 16 CPU threads, 393,216 context, 4 experts, greedy, `--serve`. The prompt is the
13K benchmark text repeated, 32,645 and 52,232 tokens.

| Prompt tokens | `--suffix-draft` | Decode | Prompt lookup |
| ---: | ---: | ---: | --- |
| 32,645 | 3 | 16 tokens, 15.18 tok/s | 8 of 12 drafts kept |
| 52,232 | 0 | 32 tokens, 8.33 tok/s | - |
| 52,232 | 3 | 32 tokens, 14.15 tok/s | 19 of 24 drafts kept |

The 52,232-token runs produced the same 32 token ids with the lookup off and on. Before `e2f2dfcee`, both long
prompts with the lookup on died after two tokens; the 52,232-token prompt with `--suffix-draft 0` ran normally.

The prose case was not measured with and without the lookup here. The policy chooses by expected value and
leaves the lookup off where a match is short or its acceptance is low; the lookup claims copied text (code
edits, quotes, templates). DSpark, when the pack has it, still drafts on novel text; the lookup is not a
replacement for it.

## Cost

- No model weights. The lookup itself uses no GPU memory; the verify buffers above are a few KB and the rollback
  rows under a megabyte a card, allocated even without a drafter.
- Host memory: the suffix index is a fixed hash table sized from the context, about 4 MiB at 32K and about
  34 MiB at 384K, for the one lookup the engine keeps.
- The proposal is a host-side hash lookup. It runs inside a verify round the engine would run anyway.

## Limits

- The lookup-alone window proposes up to 4 tokens (5 rows). Beside DSpark it can replace the drafter's window
  with up to 7 drafts (8 rows).
- It pays on copied text; on prose it usually has no match and the engine takes a plain step.
- `--lookup-chain N` extends a DSpark window; with no drafter only the lookup-alone window runs.
- The long-prompt fix was measured at 32K and 52K tokens; the same allocation covers longer prompts, but no
  separate 128K or 256K soak was run.
- Not ported: upstream `#4040 --lookup-max-t` (lookup windows up to 16 tokens, +19% on edits in Maya's notes).
  This branch's window cap is the head's 8 rows.

## Related

- [Technical reference](DETAILS.md) describes the draft layer.
- [DS4-MAYA-MERGE.md](../DS4-MAYA-MERGE.md) has the branch's other DeepSeek measurements.
