#include <optix.h>

#include "outlier_detection.h"

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
    const uint3 idx = optixGetLaunchIndex();

    float3 ray_origin;
    if constexpr (OPTIMIZATION == 0) {
        ray_origin = { float(params.points[idx.x % params.window_size].x),
                       float(params.points[idx.x % params.window_size].y),
                       float(params.points[idx.x % params.window_size].z) };
    } else if constexpr (OPTIMIZATION == 1) {
        ray_origin = { float(params.ray_origin_list[idx.x].x),
                       float(params.ray_origin_list[idx.x].y),
                       float(params.ray_origin_list[idx.x].z) };
    } else {
        ray_origin = { float(params.points[idx.x].x),
                       float(params.points[idx.x].y),
                       float(params.points[idx.x].z) };
    }
    float3 ray_direction = { 1, 0, 0 };

    unsigned int intersection_test_num = 0;
    unsigned int hit_num = 0;
    unsigned int ray_idx = idx.x % params.window_size;
    optixTrace(
            params.handle,
            ray_origin,
            ray_direction,
            params.tmin,                   // Min intersection distance
            params.tmax,                   // Max intersection distance
            0.0f,                          // rayTime -- used for motion blur
            OptixVisibilityMask( 255 ),    // Specify always visible
            OPTIX_RAY_FLAG_NONE,
            0,                             // SBT offset   -- See SBT discussion
            1,                             // SBT stride   -- See SBT discussion
            0,                             // missSBTIndex -- See SBT discussion
            intersection_test_num,
            hit_num,
            ray_idx
            );

    if constexpr (OPTIMIZATION <= 1) {
        if (hit_num <= params.K) { // include itself
            int outlier_idx = atomicAdd(params.outlier_num, 1);
            params.outlier_list[outlier_idx] = idx.x;
        }
    }

    if constexpr (DEBUG_INFO == 1) {
        params.ray_primitive_hits[idx.x] = hit_num;
        params.ray_intersections[idx.x]  = intersection_test_num;
    }
}

extern "C" __global__ void __miss__ms() {
}

extern "C" __global__ void __intersection__cube() {
    unsigned int primIdx = optixGetPrimitiveIndex();

    // OPT 2: early exit if this primitive already has enough neighbors
    if constexpr (OPTIMIZATION == 2) {
        if (params.outlier_neighbor_num[primIdx] > params.K) return;
    }

    // point  = the BVH primitive   (what we test against)
    // ray_orig = the query point    (the ray's origin)
    double3 point, ray_orig;
    if constexpr (OPTIMIZATION == 2) {
        point    = params.ray_origin_list[primIdx];
        ray_orig = params.points[optixGetPayload_2()];
    } else if constexpr (OPTIMIZATION == 1) {
        point    = params.points[primIdx];
        ray_orig = params.ray_origin_list[optixGetPayload_2()];
    } else {
        point    = params.points[primIdx];
        ray_orig = params.points[optixGetPayload_2()];
    }

    if constexpr (DEBUG_INFO == 1) {
        optixSetPayload_0(optixGetPayload_0() + 1); // number of intersection test
    }

    bool intersect = false;
    if constexpr (DIMENSION == 1) {
        if (abs(ray_orig.x - point.x) < params.R) intersect = true;
    } else if constexpr (DIMENSION == 3) {
        double3 O = { ray_orig.x - point.x, ray_orig.y - point.y, ray_orig.z - point.z };
        double sqdist = O.x * O.x + O.y * O.y + O.z * O.z;
        if (sqdist < params.R2) intersect = true;
    }

    if (intersect) {
        if constexpr (OPTIMIZATION <= 1) {
            optixSetPayload_1(optixGetPayload_1() + 1);
            if (optixGetPayload_1() > params.K) {
                optixReportIntersection(0, 0);
            }
        } else {
            if constexpr (DEBUG_INFO == 1) {
                optixSetPayload_1(optixGetPayload_1() + 1);
            }
            atomicAdd(params.outlier_neighbor_num + primIdx, 1);
        }
    }
}

extern "C" __global__ void __anyhit__terminate_ray() {
    optixTerminateRay();
}