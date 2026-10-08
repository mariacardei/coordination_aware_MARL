# Process Diagnostics

The paper's process-aware evaluation uses environment-specific diagnostics
that instantiate common coordination dimensions.

## STAT

STAT exposes assignment-level coordination directly. Important logged values
include:

- `NumConflicts_sum` and normalized conflict rates.
- `ForcedIdle_sum`, `ForcedIdleRate_per_step`, and
  `ForcedIdle_per_decision_opportunity`.
- `UniqueVictimsAssigned_mean` and assignment-diversity normalizations.
- `TaskThroughput` and `TaskCompletionRatio`.

## LBF

LBF diagnostics are implemented in `third_party/epymarl/src/envs/lbf_metrics.py`
and include:

- failed-load rate and failed-load event counts.
- successful load rate and food collection throughput.
- synchronization-failure measures.
- participation balance and partner diversity for multi-agent loading.

## RWARE

RWARE diagnostics are implemented in
`third_party/epymarl/src/envs/rware_metrics.py` and include:

- shelves delivered and delivery throughput.
- canceled movements and canceled movement rate.
- no-op rate.
- steps per successful delivery.
- per-agent delivery contribution statistics.

These diagnostics should be interpreted jointly with aggregate return. For
example, low movement contention in RWARE can reflect efficient coordination or
limited task engagement; throughput and no-op rate disambiguate the two cases.
