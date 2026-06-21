# ADR 005: Rebuild over In-Place BVH Update

**Status**: accepted
**Date**: 2024 (inferred from codebase history)

## Context

The sliding window advances by `slide` points per step. `UPDATE_GAS_TYPE == 0` performs an
in-place OptiX BVH update (only rebuilding the modified AABB regions). `UPDATE_GAS_TYPE == 1`
fully rebuilds the BVH from scratch each slide.

## Decision

Default to **rebuild** (`UPDATE_GAS_TYPE == 1`) despite the overhead.

## Rationale

1. **OPTIMIZATION == 2 changes the BVH content entirely**: When ray-BVH inversion is active,
   the BVH contains only undetermined points, whose set changes completely each slide.
   An in-place update of a totally different primitive set is effectively a rebuild anyway.

2. **Measured performance**: The OptiX BVH build is fast enough that rebuild overhead is
   negligible compared to the ray-launch phase. `OPTIX_BUILD_FLAG_PREFER_FAST_BUILD`
   trades traversal quality for build speed, which benefits the per-slide rebuild pattern.

3. **Simplicity**: One code path to maintain, test, and verify.

## When UPDATE might make sense

If `OPTIMIZATION` were 0 or 1 and the slide were very small relative to window (e.g.,
slide/window < 5%), then updating only the modified AABB regions could save build time.
However, the current workload characteristics make rebuild the better default.

## Consequences

- **Pro**: Simpler code; no need to maintain update path for edge cases
- **Pro**: Naturally compatible with OPTIMIZATION == 2
- **Con**: Rebuilds the entire BVH even when only `slide` points changed
- **Con**: `OPTIX_BUILD_FLAG_PREFER_FAST_BUILD` may produce lower-quality BVH traversal
