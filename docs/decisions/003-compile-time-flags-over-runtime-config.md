# ADR 003: Compile-Time Flags over Runtime Configuration

**Status**: accepted
**Date**: 2024 (inferred from codebase history)

## Context

The algorithm has several axes of variation (`DIMENSION`, `OPTIMIZATION`, `UPDATE_GAS_TYPE`,
`MK`, `COMPACTION`) that affect both host and device code. Device code (`outlier_detection.cu`)
uses `#if OPTIMIZATION == 0` preprocessor conditionals to select entirely different
intersection program variants.

## Decision

Use **CMake `-D` compile-time definitions** rather than runtime CLI flags for algorithm
variants. Runtime CLI flags are reserved for data-dependent parameters (R, K, window, slide).

## Why not runtime config?

Device code variants under `#if`/`#elif` cannot be toggled at runtime — they produce
different PTX entirely. Runtime branching over algorithm variants would require shipping
all variants in one binary, increasing binary size and adding runtime dispatch overhead.

Additionally, `MK` determines the static array size `int arr[MK]` in `FixQueue` (state.h),
which must be a compile-time constant in C++.

## Consequences

- **Pro**: Zero-overhead algorithm selection — dead code eliminated at compile time
- **Pro**: Natural fit for GPU code where PTX specialization matters
- **Con**: Must recompile to switch algorithm variants; experiments require multiple builds
- **Con**: `script/run.py` must call cmake + make for each parameter combination
