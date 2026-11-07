# 2026-11-07: STRATA_EXP_MODE=8 quality check (Swift 1.5 IQ3_S, 2x 2080 Ti)

Mode 8 is the expert layout that measured faster on this box (short code +7% with the daily
batch/suffix config, +20% at 100K). Before keeping it, the same greedy prompts were run with the
default Turing layout (mode 4) and with mode 8. Only `STRATA_EXP_MODE` differs; both runs use the
same engine (`0.1.57`, 384K / yarn 1.5 / KV int8), the same config (batch 2, suffix-draft 8,
spec-min-p 0.65), temperature 0 and `enable_thinking=false`.

| Case | mode 4 | mode 8 | Exact text |
|---|---|---|---|
| math: 37*43+129 | 1720, pass | 1720, pass | yes |
| logic: roses/flowers | "No", pass | "No", pass | yes |
| code: `merge_sorted` | compiles & tests pass | compiles & tests pass | yes |
| code: `is_palindrome` | compiles & tests pass | compiles & tests pass | yes |
| JSON object, 3 features | valid, pass | valid, pass | yes |
| Chinese: what is PCIe | pass | pass | yes |
| EN→ZH translation | pass | pass | yes |
| 20K needle: access code | X7-42Q, pass | X7-42Q, pass | yes |

**Result: 8/8 task checks pass in both modes; all 8 responses are byte-for-byte identical text.**
That includes the 20K needle and both generated Python functions.

This is a task-level smoke test, not a logit/KL or bitwise proof: one greedy run per prompt, eight
prompts. It does show no visible quality or output change on these tasks. The raw responses and
timings are in `quality-mode4.json` and `quality-mode8.json`.

Speed from the same branch/config (5 warm runs, short code, temp 0, thinking off):
mode 4 110.3 tok/s / 32.8 ms per decode step; mode 8 117.9 tok/s (~30.7 ms) with the daily
batch/suffix config. The single-task peak profile (no batch, suffix-draft 0, spec-min-p 0.5,
mtp-hnorm stream) reached 126.1 tok/s short code and 102.9 short list, but 100K was better with
the daily config (99.4 against 90.4).
