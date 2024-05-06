#include <optix.h>

#include "optixScan.h"

#include <sutil/vec_math.h>

extern "C" {
__constant__ Params params;
}

__forceinline__ __device__ bool operator>(const float3 a, const float3 b)
{
  return (a.x > b.x && a.y > b.y && a.z > b.z);
}

__forceinline__ __device__ bool operator<(const float3 a, const float3 b)
{
  return (a.x < b.x && a.y < b.y && a.z < b.z);
}

extern "C" __global__ void __raygen__rg() {
    // Lookup our location within the launch grid
    const uint3 idx = optixGetLaunchIndex();    
    const uint3 dim = optixGetLaunchDimensions(); 

    // Map our launch idx to a screen location and create a ray from the camera
    // location through the screen 
    float3 ray_origin, ray_direction;
#if OPTIMIZATION == 0
    ray_origin    = { float(params.points[idx.x % params.window_size].x), 
                      float(params.points[idx.x % params.window_size].y),
                      float(params.points[idx.x % params.window_size].z) };
#elif OPTIMIZATION == 1
    ray_origin    = { float(params.ray_origin_list[idx.x].x),
                      float(params.ray_origin_list[idx.x].y),
                      float(params.ray_origin_list[idx.x].z) };
#else
    ray_origin    = { float(params.points[idx.x].x), 
                      float(params.points[idx.x].y),
                      float(params.points[idx.x].z) };
#endif
    ray_direction = { 1, 0, 0 };

    // Trace the ray against our scene hierarchy
    unsigned int intersection_test_num = 0;
    unsigned int hit_num = 0;
    unsigned int ray_idx = idx.x % params.window_size;
    optixTrace(
            params.handle,
            ray_origin,
            ray_direction,
            params.tmin,                   // Min intersection distance
            params.tmax,        // Max intersection distance
            0.0f,                   // rayTime -- used for motion blur
            OptixVisibilityMask( 255 ), // Specify always visible
            OPTIX_RAY_FLAG_NONE,
            0,                   // SBT offset   -- See SBT discussion
            1,                   // SBT stride   -- See SBT discussion
            0,                   // missSBTIndex -- See SBT discussion
            intersection_test_num,
            hit_num,
            ray_idx
            );
#if OPTIMIZATION == 0 || OPTIMIZATION == 1
    if (hit_num <= params.K) { // include itself
        int outlier_idx = atomicAdd(params.outlier_num, 1);
        params.outlier_list[outlier_idx] = idx.x;
    }
#endif

#if DEBUG_INFO == 1
    params.ray_primitive_hits[idx.x] = hit_num;
    params.ray_intersections[idx.x]  = intersection_test_num;
#endif
}

extern "C" __global__ void __miss__ms() {
}

#if OPTIMIZATION == 0
extern "C" __global__ void __intersection__cube() {
    unsigned int primIdx = optixGetPrimitiveIndex();
    const double3 point = params.points[primIdx];
    const double3 ray_orig = params.points[optixGetPayload_2()]; // Get ray origin by its index
    optixSetPayload_0(optixGetPayload_0() + 1); // number of intersection test
    bool intersect = false;
    
#if DIMENSION == 1
    if (abs(ray_orig.x - point.x) < params.R) {
        intersect = true;
    }
#elif DIMENSION == 3
    // * pre AABB check 
    // bool aabb_intersect = false;
    // float3 topRight = point + params.R;
    // float3 bottomLeft = point - params.R;
    // if ((ray_orig > bottomLeft) && (ray_orig < topRight)) {
    //     aabb_intersect = true;
    // }
    // if (aabb_intersect) {
        // * exact sphere check
        double3 O = { ray_orig.x - point.x, ray_orig.y - point.y, ray_orig.z - point.z };
        double sqdist = O.x * O.x + O.y * O.y + O.z * O.z;
        if (sqdist < params.R2) {
            intersect = true;
        }
    // }
#endif

    if (intersect) {
        optixSetPayload_1(optixGetPayload_1() + 1);
        if (optixGetPayload_1() > params.K) {
            optixReportIntersection(0, 0);
        }
    }
}
#endif

#if OPTIMIZATION == 1
extern "C" __global__ void __intersection__cube() {
    unsigned int primIdx = optixGetPrimitiveIndex();
    const double3 point = params.points[primIdx];
    const double3 ray_orig = params.ray_origin_list[optixGetPayload_2()]; // Get ray origin by its index
    optixSetPayload_0(optixGetPayload_0() + 1); // number of intersection test
    bool intersect = false;
    
#if DIMENSION == 1
    if (abs(ray_orig.x - point.x) < params.R) {
        intersect = true;
    }
#elif DIMENSION == 3
    // * pre AABB check 
    // bool aabb_intersect = false;
    // float3 topRight = point + params.R;
    // float3 bottomLeft = point - params.R;
    // if ((ray_orig > bottomLeft) && (ray_orig < topRight)) {
    //     aabb_intersect = true;
    // }
    // if (aabb_intersect) {
        // * exact sphere check
        double3 O = { ray_orig.x - point.x, ray_orig.y - point.y, ray_orig.z - point.z };
        double sqdist = O.x * O.x + O.y * O.y + O.z * O.z;
        if (sqdist < params.R2) {
            intersect = true;
        }
    // }
#endif

    if (intersect) {
        optixSetPayload_1(optixGetPayload_1() + 1);
        if (optixGetPayload_1() > params.K) {
            optixReportIntersection(0, 0);
        }
    }
}
#endif

#if OPTIMIZATION == 2
extern "C" __global__ void __intersection__cube() {
    unsigned int primIdx = optixGetPrimitiveIndex();
    if (params.outlier_neighbor_num[primIdx] > params.K) return;
    const double3 point = params.ray_origin_list[primIdx];
    const double3 ray_orig = params.points[optixGetPayload_2()]; // Get ray origin by its index
#if DEBUG_INFO == 1
    optixSetPayload_0(optixGetPayload_0() + 1); // number of intersection test
#endif

    bool intersect = false;
    
#if DIMENSION == 1
    if (abs(ray_orig.x - point.x) < params.R) {
        intersect = true;
    }
#elif DIMENSION == 3
    // * pre AABB check -> no optimization
    // bool aabb_intersect = false;
    // float3 topRight = point + params.R;
    // float3 bottomLeft = point - params.R;
    // if ((ray_orig > bottomLeft) && (ray_orig < topRight)) {
    //     aabb_intersect = true;
    // }
    // if (aabb_intersect) {
        // * exact sphere check
        double3 O = { ray_orig.x - point.x, ray_orig.y - point.y, ray_orig.z - point.z };
        double sqdist = O.x * O.x + O.y * O.y + O.z * O.z;
        if (sqdist < params.R2) {
            intersect = true;
        }
    // }
#endif

    if (intersect) {
#if DEBUG_INFO == 1
        optixSetPayload_1(optixGetPayload_1() + 1);
#endif
        atomicAdd(params.outlier_neighbor_num + primIdx, 1);
    }
}
#endif

extern "C" __global__ void __anyhit__terminate_ray() {
    optixTerminateRay();
}
