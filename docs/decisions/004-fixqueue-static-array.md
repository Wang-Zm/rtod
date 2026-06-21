# ADR 004: FixQueue with Static Array Bounds (MK)

**Status**: accepted
**Date**: 2024 (inferred from codebase history)

## Context

Grid filtering assigns each point to a cell. Cells with ≤K points are "undetermined" and
need ray-based detection. The membership of each cell must be tracked, but most cells
contain far fewer than K points.

## Decision

Use a **fixed-capacity ring buffer** (`FixQueue` in `state.h`) per cell: `int arr[MK]`
with `start` and `num` tracking the ring position and count. `MK` is a compile-time constant
set via CMake (default 50, matching the typical K=50).

## Why not `std::vector` or dynamic allocation?

- Host code runs on CPU; `std::vector` would work functionally
- But millions of cells × per-cell allocations would fragment memory
- Static array avoids allocation overhead per cell
- Ring-buffer semantics naturally fit the sliding window: old points expire by decrementing `num` without shuffling array elements

## Invariant

`K <= MK` must hold at all times. This is enforced at runtime (`validate_params()` in
`outlier_detection.cpp`). If a larger K is needed, rebuild with `-D MK=<value>`.

## Consequences

- **Pro**: Zero allocation per cell, cache-friendly
- **Pro**: O(1) enqueue and dequeue (implicit via `num` decrement)
- **Con**: Must know maximum K at compile time; oversizing wastes memory
- **Con**: `MK` is a global constant — all cells share the same bound
