#include <optix.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

#include <sutil/Exception.h>
#include <sutil/sutil.h>

#include "bvh.h"

#include <iostream>

using namespace std;

void make_gas(ScanState &state) {
    size_t make_gas_start;
    start_gpu_mem(&make_gas_start);

    OptixAccelBuildOptions accel_options = {};
    if constexpr (UPDATE_GAS_TYPE == 0) {
        accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        accel_options.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    } else {
        accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
    }
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    size_t start_mem, used;
    start_gpu_mem(&start_mem);

    OptixAabb *d_aabb;
    unsigned int numPrims = state.window;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_aabb), numPrims * sizeof(OptixAabb)));
    kGenAABB(state.params.points, state.R, numPrims, d_aabb);
    state.d_aabb_ptr = reinterpret_cast<CUdeviceptr>(d_aabb);

    stop_gpu_mem(&start_mem, &used);
    std::cout << "[Mem-make_gas] kGenAABB: " << 1.0 * used / (1 << 20) << std::endl;
    start_gpu_mem(&start_mem);

    // Our build input is a simple list of non-indexed triangle vertices
    OptixBuildInput &vertex_input = state.vertex_input;
    vertex_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    vertex_input.customPrimitiveArray.aabbBuffers = &state.d_aabb_ptr;
    vertex_input.customPrimitiveArray.flags = state.vertex_input_flags;
    vertex_input.customPrimitiveArray.numSbtRecords = 1;
    vertex_input.customPrimitiveArray.numPrimitives = numPrims;
    // it's important to pass 0 to sbtIndexOffsetBuffer
    vertex_input.customPrimitiveArray.sbtIndexOffsetBuffer = 0;
    vertex_input.customPrimitiveArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
    vertex_input.customPrimitiveArray.primitiveIndexOffset = 0;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
                state.context,
                &accel_options,
                &vertex_input,
                1, // Number of build inputs
                &gas_buffer_sizes
                ));
    state.gas_buffer_sizes = gas_buffer_sizes;
    CUDA_CHECK(cudaMalloc(
               reinterpret_cast<void **>(&state.d_temp_buffer_gas),
               gas_buffer_sizes.tempSizeInBytes
              ));

    stop_gpu_mem(&start_mem, &used);
    std::cout << "[Mem-make_gas] d_temp_buffer_gas: " << 1.0 * used / (1 << 20) << std::endl;
    start_gpu_mem(&start_mem);

    // non-compacted output and size of compacted GAS.
    // CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
    size_t compactedSizeOffset = roundUp<size_t>(gas_buffer_sizes.outputSizeInBytes, 8ull);
    CUDA_CHECK(cudaMalloc(
               reinterpret_cast<void **>(&state.d_gas_output_buffer),
               compactedSizeOffset + 8
              ));

    stop_gpu_mem(&start_mem, &used);
    std::cout << "[Mem-make_gas] d_gas_output_buffer: " << 1.0 * used / (1 << 20) << std::endl;
    start_gpu_mem(&start_mem);

    size_t final_gas_size;
    OPTIX_CHECK(optixAccelBuild(
                state.context,
                0, // CUDA stream
                &accel_options,
                &vertex_input,
                1, // num build inputs
                state.d_temp_buffer_gas,
                gas_buffer_sizes.tempSizeInBytes,
                state.d_gas_output_buffer,
                gas_buffer_sizes.outputSizeInBytes,
                &state.gas_handle,
                nullptr,
                0
        ));

    stop_gpu_mem(&start_mem, &used);
    std::cout << "[Mem-make_gas] optixAccelBuild: " << 1.0 * used / (1 << 20) << std::endl;

    final_gas_size            = compactedSizeOffset;
    printf("Final GAS size: %f MB\n", (float)final_gas_size / (1024 * 1024));

    size_t make_gas_used;
    stop_gpu_mem(&make_gas_start, &make_gas_used);
    std::cout << "[Mem] make_gas: " << 1.0 * make_gas_used / (1 << 20) << std::endl;
}

void update_gas(ScanState &state, int update_pos) {
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE |
                               OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    accel_options.operation = OPTIX_BUILD_OPERATION_UPDATE;

    // update aabb
    OptixAabb *d_aabb = reinterpret_cast<OptixAabb *>(state.d_aabb_ptr);
    kGenAABB(state.params.points + update_pos * state.slide,
             state.params.R,
             state.slide,
             d_aabb + update_pos * state.slide);

    state.vertex_input.customPrimitiveArray.aabbBuffers = &state.d_aabb_ptr;
    const uint32_t vertex_input_flags[1]                = {OPTIX_GEOMETRY_FLAG_NONE};
    state.vertex_input.customPrimitiveArray.flags       = vertex_input_flags;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        state.context,
        &accel_options,
        &state.vertex_input,
        1, // Number of build inputs
        &gas_buffer_sizes));
    CUdeviceptr d_temp_update;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_temp_update), gas_buffer_sizes.tempSizeInBytes));
    OPTIX_CHECK(optixAccelBuild(
        state.context,
        0, // CUDA stream
        &accel_options,
        &state.vertex_input,
        1, // num build inputs
        d_temp_update,
        gas_buffer_sizes.tempSizeInBytes,
        state.d_gas_output_buffer,
        gas_buffer_sizes.outputSizeInBytes,
        &state.gas_handle,
        nullptr,
        0
        ));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_temp_update)));
}

void rebuild_gas(ScanState &state, int update_pos) {
    if (state.params.ray_origin_num == 0) {
        return;
    }

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_BUILD; // * bring higher performance compared to OPTIX_BUILD_FLAG_PREFER_FAST_TRACE
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    // update aabb
    OptixAabb *d_aabb = reinterpret_cast<OptixAabb *>(state.d_aabb_ptr);
    if constexpr (OPTIMIZATION <= 1) {
        kGenAABB(state.params.points + update_pos * state.slide,
                 state.params.R,
                 state.slide,
                 d_aabb + update_pos * state.slide);
        state.vertex_input.customPrimitiveArray.numPrimitives = state.window;
    } else {
        kGenAABB(state.params.ray_origin_list,
                 state.params.R,
                 state.params.ray_origin_num,
                 d_aabb);
        state.vertex_input.customPrimitiveArray.numPrimitives = state.params.ray_origin_num;
    }

    // recompute gas_buffer_sizes
    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
                state.context,
                &accel_options,
                &state.vertex_input,
                1, // Number of build inputs
                &gas_buffer_sizes
                ));
    // invariant: the temp buffer allocated by make_gas (sized for window)
    // must be large enough for this rebuild (sized for undetermined points)
    assert(gas_buffer_sizes.tempSizeInBytes <= state.gas_buffer_sizes.tempSizeInBytes);
    assert(gas_buffer_sizes.outputSizeInBytes <= state.gas_buffer_sizes.outputSizeInBytes);
    OPTIX_CHECK(optixAccelBuild(
                state.context,
                0, // CUDA stream
                &accel_options,
                &state.vertex_input,
                1, // num build inputs
                state.d_temp_buffer_gas,
                gas_buffer_sizes.tempSizeInBytes,
                state.d_gas_output_buffer,
                gas_buffer_sizes.outputSizeInBytes,
                &state.gas_handle,
                nullptr,
                0
        ));
}