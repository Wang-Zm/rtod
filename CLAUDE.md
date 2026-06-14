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
| `DIMENSION` | Data dimensionality | `1` or `3` |
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
│   ├── outlier_detection.cpp        # Host code: init OptiX, build GAS/pipeline/SBT, sliding window loop
│   ├── outlier_detection.cu         # Device code: raygen, intersection, miss, anyhit programs
│   ├── aabb.cu             # GPU kernel kGenAABB: computes per-point AABBs for BVH
│   ├── state.h             # ScanState (all state) + FixQueue (ring buffer for grid cells)
│   └── timer.h             # High-precision timer for per-phase profiling
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

## Key Design Choices

- **Custom primitives** (not triangles): The BVH contains spheres (points). `outlier_detection.cu` defines a custom intersection program that tests for distance-based proximity rather than geometric intersection.
- **FixQueue** (`state.h`): A fixed-capacity ring buffer (`MK` slots) used for grid cell membership tracking. Each cell holds up to MK points.
- **OptiX pipeline**: Single raygen program + single miss program + one hitgroup (intersection + anyhit). The anyhit terminates the ray once K+1 neighbors are found (early exit).
- **Warmup**: 10 warmup slides run before timed slides to stabilize GPU clocks and caches.
- **`DEBUG_INFO`**: When set to 1, collects per-ray hit/intersection counts for analysis. Set to 0 for production benchmarking.
