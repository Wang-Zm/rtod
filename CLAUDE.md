# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RTOD (Real-Time Outlier Detection) uses NVIDIA OptiX hardware ray tracing (RTX) on GPUs to detect distance-based outliers in streaming data. It evaluates sliding windows with a novel BVH-based approach and compares against MCOD, NETS, and MDUAL baselines.

## Build & Run

```bash
# Build from src/ into build/
cd build/ && cmake ../src/ -D DIMENSION=1 -D MK=50 \
    -D COMPACTION=0 -D UPDATE_GAS_TYPE=1 \
    -D CMAKE_BUILD_TYPE=Release -D OPTIMIZATION=2 && make

# Run a scan
./build/bin/outlier_detection --n 1000000 --R 0.028 --K 50 --window 100000 --slide 5000 -f data/gaussian.txt
```

**Compile-time flags** (passed via CMake `-D`):

| Flag | Meaning | Values |
|------|---------|--------|
| `DIMENSION` | Data dimensionality | `1`, `2`, or `3` |
| `MK` | Max K value (array sizing) | e.g. `50` |
| `OPTIMIZATION` | Algorithm variant | `0`=baseline (all-points), `1`=grid filtering, `2`=grid filtering + ray-BVH inversion (default) |
| `UPDATE_GAS_TYPE` | BVH update strategy | `0`=in-place update, `1`=rebuild |
| `COMPACTION` | BVH compaction | `0`=off, `1`=on |
| `CMAKE_BUILD_TYPE` | Build config | `Release` or `Debug` |

**Runtime flags** for `outlier_detection`: `--n`, `--R`, `--K`, `--window`, `--slide`, `-f <datafile>`, `--start_copy_pos`, `--launch_ray_num`.

## Running Experiments

```bash
python script/run.py   # orchestrates cmake + make + outlier_detection for all dataset/parameter combos
python script/draw.py  # generates paper figures (PDFs) from hardcoded experiment data
```

**Hardware requirement**: NVIDIA RTX GPU (2000/3000/4000 series or RTX-capable workstation/datacenter GPU). Requires NVIDIA OptiX SDK (7.x) and CUDA Toolkit.

## Architecture

```
src/
├── CMakeLists.txt          # Top-level build; finds CUDA + OptiX, sets flags
├── sampleConfig.h.in       # Template for sample-wide config
├── CMake/                  # FindCUDA, FindOptiX, compiler flags, macros
├── sutil/                  # OptiX SDK utility library (AABB, Camera, math)
├── ods/                    # Main application (outlier detection on streams)
│   ├── CMakeLists.txt               # Defines the outlier_detection executable target
│   ├── outlier_detection.h          # Params struct (device-side constants), Data structs
│   ├── outlier_detection.cpp        # Main: arg parsing, data I/O, mem alloc, sliding-window loop
│   ├── pipeline.h / pipeline.cpp    # OptiX pipeline lifecycle (context, module, programs, SBT)
│   ├── bvh.h / bvh.cpp              # BVH/GAS build, update, and rebuild
│   ├── grid.h / grid.cpp            # Grid filtering: cell assignment, undetermined-point selection
│   ├── verify.h / verify.cpp        # Debug verification (DEBUG_INFO only): result checking, stats
│   ├── outlier_detection.cu         # Device code: raygen, intersection, miss, anyhit programs
│   ├── aabb.cu                      # GPU kernel kGenAABB: computes per-point AABBs for BVH
│   ├── state.h                      # ScanState (all state) + FixQueue (ring buffer for grid cells)
│   └── timer.h                      # High-precision timer for per-phase profiling
include/                    # OptiX 7 API headers (optix.h, optix_device.h, etc.)
data/                       # Input datasets: gaussian.txt (1D), stock.txt (1D), tao.txt (3D)
script/
├── run.py                  # Experiment runner: builds with cmake flags, launches outlier_detection
└── draw.py                 # Matplotlib figure generation for paper plots
pic/                        # Output PDF figures
```

## Core Algorithm (per sliding window)

1. **Data transfer**: Copy new slide batch H→D (`cudaMemcpy`)
2. **Grid filtering** (OPT ≥ 1): Assign points to grid cells; only points in cells with ≤K members are "undetermined" and need ray-based detection
3. **BVH build/update**: Build or update AABB BVH over points via `optixAccelBuild`
4. **Outlier detection** (`optixLaunch`): For each candidate point, cast a ray; the intersection program increments a neighbor counter when a neighbor is within distance R. Points with ≤K neighbors are outliers.
5. **Result transfer**: Copy outlier indices D→H

**Optimization level 2** inverts the traditional approach: instead of casting rays FROM candidate points and testing against all points in BVH, it builds a BVH over only the undetermined points and casts rays from ALL window points, atomically incrementing neighbor counters on the BVH primitives.

## Harness — Verification Gates

Run all checks before committing:

```bash
bash script/check.sh
```

This enforces 5 gates (see script for details):
- **Gate A**: File existence — every file CLAUDE.md references still exists
- **Gate B**: Compile-time invariant guards — `#error` directives present in `outlier_detection.h` and CMake-level checks in `ods/CMakeLists.txt`
- **Gate C**: Runtime invariant guards — `validate_params()` called from `main`, guards for `window % slide`, `K <= MK`
- **Gate D**: ADR coverage — all 5 expected ADRs present in `docs/decisions/`
- **Gate E**: Forbidden patterns — no `using namespace std` in headers

## Invariants

### Compile-time (enforced via `#error` in `outlier_detection.h` + CMake `FATAL_ERROR`)

| Invariant | Location | Rationale |
|-----------|----------|-----------|
| `DIMENSION ∈ {1, 2, 3}` | `outlier_detection.h` + `ods/CMakeLists.txt` | Dimension-specific distance formulas in device code |
| `OPTIMIZATION ∈ {0, 1, 2}` | both | Each value selects a different intersection program in `.cu` |
| `UPDATE_GAS_TYPE ∈ {0, 1}` | both | OptiX only supports build or update operations |
| `COMPACTION == 0` | `outlier_detection.h` | BVH compaction path not implemented |
| `MK >= 1` | `outlier_detection.h` | FixQueue uses `arr[MK]` |

### Runtime (enforced via `validate_params()` + `assert` in `outlier_detection.cpp`)

| Invariant | Check | Consequence if violated |
|-----------|-------|------------------------|
| `K <= MK` | `validate_params()` | FixQueue overflow → segfault |
| `window % slide == 0` | `validate_params()` | Ring-buffer index corruption |
| `data_num >= window` | `validate_params()` | Out-of-bounds memory access |
| `window > 0, slide > 0, R > 0, K > 0` | `validate_params()` | Undefined behavior |
| `ray_origin_num <= window_size` | `assert` after grid filtering | Buffer overflow in `h_ray_origin_list` |
| `outlier_num <= window_size` | `assert` after each slide | Logic error in outlier counting |

## Architecture Decision Records (ADRs)

Key design decisions are documented in `docs/decisions/`. Read these before proposing changes to the core algorithm:

| ADR | Topic |
|-----|-------|
| [001](docs/decisions/001-custom-primitives-instead-of-triangles.md) | Custom primitives instead of triangles for point-distance queries |
| [002](docs/decisions/002-ray-bvh-inversion.md) | Ray-BVH inversion (OPTIMIZATION == 2) |
| [003](docs/decisions/003-compile-time-flags-over-runtime-config.md) | Compile-time flags over runtime configuration |
| [004](docs/decisions/004-fixqueue-static-array.md) | FixQueue with static array bounds (MK) |
| [005](docs/decisions/005-rebuild-over-update.md) | Rebuild over in-place BVH update |

## Forbidden Patterns

- **No `using namespace std` in headers** — pollutes every includer's namespace. Qualify with `std::`.
- **No malloc for GPU-visible memory** — always use `cudaMalloc` / `cudaFree`.
- **No hardcoded `/home/wzm/` paths** — use relative paths or configurable directories.
- **Don't add runtime CLI flags for algorithm variants** — those belong in compile-time `if constexpr` blocks (see [ADR 003](docs/decisions/003-compile-time-flags-over-runtime-config.md)).
- **After code changes, review and update CLAUDE.md and docs/decisions/** — if a code change alters a fact documented in these files (DIMENSION range, C++ standard, invariants, file structure), update the docs in the same commit. Gate F in `script/check.sh` catches some drifts automatically, but not all.

## Key Design Choices

- **Custom primitives** (not triangles): The BVH contains spheres (points). `outlier_detection.cu` defines a custom intersection program that tests for distance-based proximity rather than geometric intersection.
- **FixQueue** (`state.h`): A fixed-capacity ring buffer (`MK` slots) used for grid cell membership tracking. Each cell holds up to MK points.
- **OptiX pipeline**: Single raygen program + single miss program + one hitgroup (intersection + anyhit). The anyhit terminates the ray once K+1 neighbors are found (early exit).
- **Warmup**: 10 warmup slides run before timed slides to stabilize GPU clocks and caches.
- **`DEBUG_INFO`**: When set to 1, collects per-ray hit/intersection counts for analysis. Set to 0 for production benchmarking.
