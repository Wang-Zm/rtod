#ifndef OUTLIER_DETECTION_H
#define OUTLIER_DETECTION_H

#ifndef DATA_N
#define DATA_N  1e8
#endif

#define DEBUG_INFO 0

#ifndef DIMENSION
#define DIMENSION 1
#endif

#ifndef COMPACTION
#define COMPACTION 0
#endif

#ifndef UPDATE_GAS_TYPE
#define UPDATE_GAS_TYPE 1 // 0: update, 1: rebuild
#endif

#ifndef OPTIMIZATION
#define OPTIMIZATION 2 // 0: No Opt, 1: only Grid Filtering, 2: Grid Filtering + Ray-BVH Inversing
#endif

#ifndef MK
#define MK 50
#endif

// ── compile-time invariants ─────────────────────────────────────────────
// These guards prevent misconfiguration at build time.
// If you hit one of these, fix the CMake flags (-D...) and reconfigure.

#if DIMENSION != 1 && DIMENSION != 3
#error "DIMENSION must be 1 or 3"
#endif

#if OPTIMIZATION < 0 || OPTIMIZATION > 2
#error "OPTIMIZATION must be 0, 1, or 2"
#endif

#if UPDATE_GAS_TYPE != 0 && UPDATE_GAS_TYPE != 1
#error "UPDATE_GAS_TYPE must be 0 (update) or 1 (rebuild)"
#endif

#if COMPACTION != 0
#error "COMPACTION != 0 is not supported (BVH compaction not implemented)"
#endif

#if MK < 1
#error "MK must be >= 1"
#endif

struct Params
{
    double3*                points;    
    OptixTraversableHandle  handle;
    
    float                   tmin;
    float                   tmax;
    unsigned int*           intersection_test_num;
    unsigned*               hit_num;
    double                  R;
    int                     K;

    int*                    outlier_list;           // store the index of outliers in the current window
    int*                    outlier_num;
    int*                    outlier_neighbor_num;   // store the number of neighbors of each outlier
    
    unsigned*               ray_primitive_hits;
    unsigned*               ray_intersections;

    double3*                ray_origin_list;
    int                     ray_origin_num;
    double                  R2;
    int                     window_size;
};


struct RayGenData
{
    // No data needed
};


struct MissData
{
};


struct HitGroupData
{
};

#endif