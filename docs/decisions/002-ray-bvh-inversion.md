# ADR 002: Ray-BVH Inversion (OPTIMIZATION == 2)

**Status**: accepted
**Date**: 2024 (inferred from codebase history)

## Context

OPTIMIZATION levels 0 and 1 cast rays **from** candidate (undetermined) points **into** a
BVH containing all window points. When the fraction of undetermined points is high, this
is efficient. But when only a small fraction of points are undetermined (common with
effective grid filtering), we waste BVH capacity on determined points that don't need
ray-casting.

## Decision

**Invert the ray↔BVH relationship**: build the BVH over only the **undetermined** points,
and cast rays **from all window points** into that much smaller BVH. Use `atomicAdd` on
the BVH primitive's neighbor counter to increment from multiple simultaneous rays.

## Mechanics

1. Grid filtering identifies undetermined points (cells with ≤K members)
2. BVH is built over only these undetermined points (typically << window)
3. All window points cast rays into the undetermined BVH
4. Each ray reports `optixReportIntersection` upon finding a valid neighbor
5. Intersection handler atomically increments `outlier_neighbor_num[primIdx]`
6. After launch, points with `outlier_neighbor_num[i] <= K` are outliers

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| OPTIMIZATION == 1 (grid filter, forward rays) | BVH still contains all window points; wastes capacity on determined points |
| CUDA brute-force on only undetermined points | No hardware BVH acceleration |

## Consequences

- **Pro**: BVH size dramatically smaller when < 10% of points are undetermined
- **Pro**: Leverages RTX hardware for massively parallel neighbor counting via atomics
- **Con**: Different intersection program for each `OPTIMIZATION` value (code duplication in `outlier_detection.cu`)
- **Con**: Atomic contention possible when many points neighbor the same undetermined point
