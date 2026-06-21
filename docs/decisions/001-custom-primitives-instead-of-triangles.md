# ADR 001: Custom Primitives Instead of Triangles

**Status**: accepted
**Date**: 2024 (inferred from codebase history)

## Context

OptiX primarily targets triangle-mesh ray tracing for rendering. However, RTOD's domain is
distance-based outlier detection on point data — each data point is a sphere of radius R,
not a triangle.

## Decision

Use **`OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES`** with a hand-written intersection program
(`__intersection__cube` in `outlier_detection.cu`) that tests the distance between the ray
origin and each BVH primitive (sphere/point), rather than geometric ray-triangle intersection.

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| Triangles (default OptiX path) | Point data has no triangles; synthesizing them would add overhead and complexity without benefit |
| AABB-only intersection | Too coarse — would produce false positives (points within bounding box but outside radius R) |
| CUDA brute-force (no BVH) | O(n²) per window; BVH gives O(log n) with hardware acceleration |

## Consequences

- **Pro**: Leverages NVIDIA RTX hardware BVH traversal for point-distance queries
- **Pro**: Early termination via `optixReportIntersection` / `optixTerminateRay` once K+1 neighbors found
- **Con**: Must author custom intersection program per `OPTIMIZATION` variant
- **Con**: Cannot use OptiX built-in triangle intersection unit; custom intersection runs in software
