#ifndef OPTIXSCAN_H
#define OPTIXSCAN_H

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

#define MK 50

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