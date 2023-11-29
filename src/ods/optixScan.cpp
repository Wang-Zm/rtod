#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

#include <sampleConfig.h>

#include <sutil/Exception.h>
#include <sutil/sutil.h>

#include "state.h"
#include "timer.h"

#include <array>
#include <bitset>
#include <iomanip>
#include <iostream>
#include <string>
#include <unistd.h>
#include <map>
#include <thread>
#include <sutil/Camera.h>

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>

#include <fstream>
using namespace std;

template <typename T>
struct SbtRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData> RayGenSbtRecord;
typedef SbtRecord<MissData> MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;

int                      data_num = DATA_N; 
double3*                 vertices;
Timer                    timer;

extern "C" void kGenAABB(double3 *points, double radius, unsigned int numPrims, OptixAabb *d_aabb);

void printUsageAndExit(const char* argv0) {
    std::cerr << "Usage  : " << argv0 << " [options]\n";
    std::cerr << "Options: --file | -f <filename>      Specify file for data input\n";
    std::cerr << "         --help | -h                 Print this usage message\n";
    std::cerr << "         --n <int>                   Set data num; defaults to 1e8\n";
    std::cerr << "         --primitive <int>           Set primitive type, 0 for cube, 1 for triangle with anyhit; defaults to 0\n";
    std::cerr << "         --nc                        No Comparison\n";
    exit(1);
}

static void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */) {
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: "
              << message << "\n";
}

size_t get_cpu_memory_usage() {
    FILE* file = fopen("/proc/self/status", "r");
    int result = -1;
    char line[128];
    while (fgets(line, 128, file) != nullptr) {
        if (strncmp(line, "VmRSS:", 6) == 0) {
            int len = strlen(line);
            const char* p = line;
            for (; std::isdigit(*p) == false; ++p) {}
            line[len - 3] = 0;
            result = atoi(p);
            break;
        }
    }
    fclose(file);
    return result; // KB
}

void start_gpu_mem(size_t* avail_mem) {
    size_t total_gpu_mem;
    CUDA_CHECK(cudaMemGetInfo( avail_mem, &total_gpu_mem ));
}

void stop_gpu_mem(size_t* avail_mem, size_t* used) {
    size_t total_gpu_mem, avail_mem_now;
    CUDA_CHECK(cudaMemGetInfo( &avail_mem_now, &total_gpu_mem ));
    *used = *avail_mem - avail_mem_now;
}

void read_data(std::string& outfile, ScanState &state) {
    vertices = (double3*) malloc(data_num * sizeof(double3));

    ifstream fin;
    string line;
    fin.open(outfile, ios::in);
    if (!fin.is_open()) {
        cerr << "Fail to open [" << outfile << "]!" << endl;
    }
    for (int dim_id = 0; dim_id < DIMENSION; dim_id++) {
        state.max_value[dim_id] = -FLT_MAX;
        state.min_value[dim_id] = FLT_MAX;
    }
    for (int rid = 0; rid < data_num; rid++) {
        getline(fin, line);
#if DIMENSION == 1
        sscanf(line.c_str(), "%lf", &vertices[rid].x);
        vertices[rid].y = vertices[rid].z = 0;
        if (state.max_value[0] < vertices[rid].x) {
            state.max_value[0] = vertices[rid].x;
        }
        if (state.min_value[0] > vertices[rid].x) {
            state.min_value[0] = vertices[rid].x;
        }
#elif DIMENSION == 3
        sscanf(line.c_str(), "%lf,%lf,%lf", &vertices[rid].x, &vertices[rid].y, &vertices[rid].z);
        if (state.max_value[0] < vertices[rid].x) {
            state.max_value[0] = vertices[rid].x;
        }
        if (state.min_value[0] > vertices[rid].x) {
            state.min_value[0] = vertices[rid].x;
        }

        if (state.max_value[1] < vertices[rid].y) {
            state.max_value[1] = vertices[rid].y;
        }
        if (state.min_value[1] > vertices[rid].y) {
            state.min_value[1] = vertices[rid].y;
        }

        if (state.max_value[2] < vertices[rid].z) {
            state.max_value[2] = vertices[rid].z;
        }
        if (state.min_value[2] > vertices[rid].z) {
            state.min_value[2] = vertices[rid].z;
        }
#endif
    }

    for (int i = 0; i < DIMENSION; i++) {
        std::cout << "DIM[" << i << "]: " << state.min_value[i] << ", " << state.max_value[i] << std::endl;
    }
}

void parse_args(ScanState &state, int argc, char *argv[]) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--help" || arg == "-h") {
            printUsageAndExit(argv[0]);
        } else if (arg == "--file" || arg == "-f") {
            if (i < argc - 1) {
                state.infile = argv[++i];
            } else {
                printUsageAndExit(argv[0]);
            }
        } else if (arg == "--n") {
            if (i < argc - 1) {
                data_num = stoi(argv[++i]);
            } else {
                printUsageAndExit(argv[0]);
            }
        } else if(arg == "--window") {
            if (i < argc - 1) {
                state.window = stoi(argv[++i]);
            } else {
                printUsageAndExit(argv[0]);
            }
        } else if(arg == "--slide"){
            if (i < argc - 1) {
                state.slide = stoi(argv[++i]);
            } else {
                printUsageAndExit(argv[0]);
            }
        }
        else if (arg == "--R") {
            if (i < argc - 1) {
                state.R = stod(argv[++i]);
            } else {
                printUsageAndExit(argv[0]);
            }
        }
        else if (arg == "--K") {
            if (i < argc - 1) {
                state.K = stoi(argv[++i]);
            } else {
                printUsageAndExit(argv[0]);
            }
        }
        else if (arg == "--start_copy_pos") {
            if (i < argc - 1) {
                state.start_copy_pos = stoi(argv[++i]);
            } else {
                printUsageAndExit(argv[0]);
            }
        }
        else if (arg == "--launch_ray_num") {
            if (i < argc - 1) {
                state.launch_ray_num = stoi(argv[++i]);
            } else {
                printUsageAndExit(argv[0]);
            }
        }
        else {
            std::cerr << "Unknown option '" << arg << "'\n";
            printUsageAndExit(argv[0]);
        }
    }
}

void initialize_optix(ScanState &state) {
    // Initialize CUDA
    CUDA_CHECK(cudaFree(0));

    // Initialize the OptiX API, loading all API entry points
    OPTIX_CHECK(optixInit());

    // Specify context options
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &context_log_cb;
    options.logCallbackLevel = 4;

    // Associate a CUDA context (and therefore a specific GPU) with this
    // device context
    CUcontext cuCtx = 0; // zero means take the current context
    OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &state.context));
}

void data_h2d(ScanState &state) {
    size_t start, used;
    start_gpu_mem(&start);
    CUDA_CHECK(cudaMalloc(&state.params.points, state.window * sizeof(double3)));
    stop_gpu_mem(&start, &used);
    std::cout << "[Mem] data_h2d: " << 1.0 * used / (1 << 20) << std::endl;
}

void make_gas(ScanState &state) {
    size_t make_gas_start;
    start_gpu_mem(&make_gas_start);

    OptixAccelBuildOptions accel_options = {};
#if UPDATE_GAS_TYPE == 0
    accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    accel_options.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_UPDATE;
#else
    accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
#endif
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
#if OPTIMIZATION == 0 || OPTIMIZATION == 1
    kGenAABB(state.params.points + update_pos * state.slide, 
             state.params.R, 
             state.slide, 
             d_aabb + update_pos * state.slide);
    state.vertex_input.customPrimitiveArray.numPrimitives = state.window;
#else 
    kGenAABB(state.params.ray_origin_list, 
             state.params.R, 
             state.params.ray_origin_num, 
             d_aabb);
    state.vertex_input.customPrimitiveArray.numPrimitives = state.params.ray_origin_num;
#endif

    // recompute gas_buffer_sizes
    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
                state.context,
                &accel_options,
                &state.vertex_input,
                1, // Number of build inputs
                &gas_buffer_sizes
                ));
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

void make_module(ScanState &state) {
    size_t make_module_start_mem;
    start_gpu_mem(&make_module_start_mem);

    char log[2048];

    OptixModuleCompileOptions module_compile_options = {};
#if !defined(NDEBUG)
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif

    state.pipeline_compile_options.usesMotionBlur = false;
    state.pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    state.pipeline_compile_options.numPayloadValues = 3;
    state.pipeline_compile_options.numAttributeValues = 0;
#ifdef DEBUG // Enables debug exceptions during optix launches. This may incur significant performance cost and should only be done during development.
    pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#else
    state.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
    state.pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
    // By default (usesPrimitiveTypeFlags == 0) it supports custom and triangle primitives
    state.pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;

    size_t inputSize = 0;
    const char *input = sutil::getInputData(OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "optixScan.cu", inputSize);
    size_t sizeof_log = sizeof(log);

    OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
        state.context,
        &module_compile_options,
        &state.pipeline_compile_options,
        input,
        inputSize,
        log,
        &sizeof_log,
        &state.module));
    
    size_t used;
    stop_gpu_mem(&make_module_start_mem, &used);
    std::cout << "[Mem] make_module: " << 1.0 * used / (1 << 20) << std::endl;
}

void make_program_groups(ScanState &state) {
    size_t start;
    start_gpu_mem(&start);

    char log[2048];

    OptixProgramGroupOptions program_group_options = {}; // Initialize to zeros

    OptixProgramGroupDesc raygen_prog_group_desc = {};
    raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module = state.module;
    raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        state.context,
        &raygen_prog_group_desc,
        1, // num program groups
        &program_group_options,
        log,
        &sizeof_log,
        &state.raygen_prog_group));

    OptixProgramGroupDesc miss_prog_group_desc = {};
    miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module = state.module;
    miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
    sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        state.context,
        &miss_prog_group_desc,
        1, // num program groups
        &program_group_options,
        log,
        &sizeof_log,
        &state.miss_prog_group));

    OptixProgramGroupDesc hitgroup_prog_group_desc = {};
    hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_prog_group_desc.hitgroup.moduleIS = state.module;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__cube";
    hitgroup_prog_group_desc.hitgroup.moduleAH = state.module;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__terminate_ray";
      
    sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        state.context,
        &hitgroup_prog_group_desc,
        1, // num program groups
        &program_group_options,
        log,
        &sizeof_log,
        &state.hitgroup_prog_group));

    size_t used;
    stop_gpu_mem(&start, &used);
    std::cout << "[Mem] make_program_groups: " << 1.0 * used / (1 << 20) << std::endl;
}

void make_pipeline(ScanState &state) {
    size_t start;
    start_gpu_mem(&start);

    char log[2048];
    const uint32_t max_trace_depth = 1;
    std::vector<OptixProgramGroup> program_groups{state.raygen_prog_group, state.miss_prog_group, state.hitgroup_prog_group};

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = max_trace_depth;
    pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixPipelineCreate(
        state.context,
        &state.pipeline_compile_options,
        &pipeline_link_options,
        program_groups.data(),
        program_groups.size(),
        log,
        &sizeof_log,
        &state.pipeline));

    OptixStackSizes stack_sizes = {};
    for (auto &prog_group : program_groups) {
        OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes));
    }

    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, max_trace_depth,
                                           0, // maxCCDepth
                                           0, // maxDCDEpth
                                           &direct_callable_stack_size_from_traversal,
                                           &direct_callable_stack_size_from_state, &continuation_stack_size));
    OPTIX_CHECK(optixPipelineSetStackSize(state.pipeline, direct_callable_stack_size_from_traversal,
                                          direct_callable_stack_size_from_state, continuation_stack_size,
                                          1 // maxTraversableDepth
                                          ));
    
    size_t used;
    stop_gpu_mem(&start, &used);
    std::cout << "[Mem] make_pipeline: " << 1.0 * used / (1 << 20) << std::endl;
}

void make_sbt(ScanState &state) {
    size_t start;
    start_gpu_mem(&start);

    CUdeviceptr raygen_record;
    const size_t raygen_record_size = sizeof(RayGenSbtRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&raygen_record), raygen_record_size));
    RayGenSbtRecord rg_sbt;
    OPTIX_CHECK(optixSbtRecordPackHeader(state.raygen_prog_group, &rg_sbt));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>(raygen_record),
        &rg_sbt,
        raygen_record_size,
        cudaMemcpyHostToDevice));

    CUdeviceptr miss_record;
    size_t miss_record_size = sizeof(MissSbtRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&miss_record), miss_record_size));
    MissSbtRecord ms_sbt;
    OPTIX_CHECK(optixSbtRecordPackHeader(state.miss_prog_group, &ms_sbt));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>(miss_record),
        &ms_sbt,
        miss_record_size,
        cudaMemcpyHostToDevice));

    CUdeviceptr hitgroup_record;
    size_t hitgroup_record_size = sizeof(HitGroupSbtRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&hitgroup_record), hitgroup_record_size));
    HitGroupSbtRecord hg_sbt;
    OPTIX_CHECK(optixSbtRecordPackHeader(state.hitgroup_prog_group, &hg_sbt));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>(hitgroup_record),
        &hg_sbt,
        hitgroup_record_size,
        cudaMemcpyHostToDevice));

    state.sbt.raygenRecord = raygen_record;
    state.sbt.missRecordBase = miss_record;
    state.sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
    state.sbt.missRecordCount = 1;
    state.sbt.hitgroupRecordBase = hitgroup_record;
    state.sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
    state.sbt.hitgroupRecordCount = 1;

    size_t used;
    stop_gpu_mem(&start, &used);
    std::cout << "[Mem] make_sbt: " << 1.0 * used / (1 << 20) << std::endl;
}

// void initialize_params(ScanState &state) {
//     size_t start;
//     start_gpu_mem(&start);

// #if OPTIMIZATION == 1 || OPTIMIZATION == 2
//     state.h_current_window = (double3 *) malloc(state.window * sizeof(double3));
    
//     state.h_ray_origin_list = (double3 *) malloc(state.window * sizeof(double3));
//     CUDA_CHECK(cudaMalloc(&state.params.ray_origin_list, state.window * sizeof(double3)));
// #endif

// #if OPTIMIZATION == 2
//     state.h_ray_origin_idx  = (int *) malloc(state.window * sizeof(int));
// #endif

//     state.h_outlier_list = (int *) malloc(state.window * sizeof(int));
// #if OPTIMIZATION == 0 || OPTIMIZATION == 1
//     CUDA_CHECK(cudaMalloc(&state.params.outlier_list, state.window * sizeof(int)));
// #endif

//     CUDA_CHECK(cudaMalloc(&state.params.outlier_num, sizeof(int)));

// #if OPTIMIZATION == 2
//     CUDA_CHECK(cudaMalloc(&state.params.outlier_neighbor_num, state.window * sizeof(int)));
//     state.h_outlier_neightbor_num = (int *) malloc(state.window * sizeof(int));
// #endif

// #if DEBUG_INFO == 1
//     CUDA_CHECK(cudaMalloc(&state.params.ray_primitive_hits, state.window * sizeof(unsigned)));
//     CUDA_CHECK(cudaMalloc(&state.params.ray_intersections, state.window * sizeof(unsigned)));

//     state.h_ray_hits = (unsigned *) malloc(state.window * sizeof(unsigned));
//     state.h_ray_intersections = (unsigned *) malloc(state.window * sizeof(unsigned));
// #endif

//     CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state.d_params), sizeof(Params)));

//     state.params.R  = state.R;
//     state.params.R2 = state.R * state.R;
//     state.params.K  = state.K;
//     state.params.handle = state.gas_handle;
//     state.params.tmin   = 0.0f;
//     state.params.tmax   = FLT_MIN;

//     size_t used;
//     stop_gpu_mem(&start, &used);
//     std::cout << "[Mem] initialize_params: " << 1.0 * used / (1 << 20) << std::endl;
// }

void initialize_params(ScanState &state) {
    size_t start;
    start_gpu_mem(&start);

    CUDA_CHECK(cudaMalloc(&state.params.outlier_num, sizeof(int)));

#if OPTIMIZATION == 1 || OPTIMIZATION == 2
    CUDA_CHECK(cudaMalloc(&state.params.ray_origin_list, state.window * sizeof(double3)));
    state.h_ray_origin_list = (double3 *) malloc(state.window * sizeof(double3));
#endif

#if OPTIMIZATION == 2
    state.h_ray_origin_idx  = (int *) malloc(state.window * sizeof(int));
#endif

#if OPTIMIZATION == 2
    state.h_outlier_neightbor_num = (int *) malloc(state.window * sizeof(int));
    CUDA_CHECK(cudaMalloc(&state.params.outlier_neighbor_num, state.window * sizeof(int)));
#endif

    state.h_outlier_list = (int *) malloc(state.window * sizeof(int));
#if OPTIMIZATION == 0 || OPTIMIZATION == 1
    CUDA_CHECK(cudaMalloc(&state.params.outlier_list, state.window * sizeof(int)));
#endif

#if OPTIMIZATION == 1 || OPTIMIZATION == 2
    state.h_current_window = (double3 *) malloc(state.window * sizeof(double3));
#endif

#if DEBUG_INFO == 1
    CUDA_CHECK(cudaMalloc(&state.params.ray_primitive_hits, state.window * sizeof(unsigned)));
    CUDA_CHECK(cudaMalloc(&state.params.ray_intersections, state.window * sizeof(unsigned)));

    state.h_ray_hits = (unsigned *) malloc(state.window * sizeof(unsigned));
    state.h_ray_intersections = (unsigned *) malloc(state.window * sizeof(unsigned));
#endif

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state.d_params), sizeof(Params)));

    state.params.R  = state.R;
    state.params.R2 = state.R * state.R;
    state.params.K  = state.K;
    state.params.handle = state.gas_handle;
    state.params.tmin   = 0.0f;
    state.params.tmax   = FLT_MIN;

    size_t used;
    stop_gpu_mem(&start, &used);
    std::cout << "[Mem] initialize_params: " << 1.0 * used / (1 << 20) << std::endl;
}


void launch(ScanState &state) {
    if (state.params.ray_origin_num == 0) {
        return;
    }

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>(state.d_params),
        &state.params,
        sizeof(Params),
        cudaMemcpyHostToDevice));

#if OPTIMIZATION == 0
    if (state.launch_ray_num != 0) {
        OPTIX_CHECK(optixLaunch(state.pipeline, 0, state.d_params, sizeof(Params), &state.sbt, state.launch_ray_num, 1, 1));
    } else {
        OPTIX_CHECK(optixLaunch(state.pipeline, 0, state.d_params, sizeof(Params), &state.sbt, state.window, 1, 1));
    }
#elif OPTIMIZATION == 1
    OPTIX_CHECK(optixLaunch(state.pipeline, 0, state.d_params, sizeof(Params), &state.sbt, state.params.ray_origin_num, 1, 1));
#else
    OPTIX_CHECK(optixLaunch(state.pipeline, 0, state.d_params, sizeof(Params), &state.sbt, state.window, 1, 1));
#endif
    CUDA_SYNC_CHECK();
}

void result_d2h(ScanState &state, int outlier_num, int window_id, int unit_num) {
    CUDA_CHECK(cudaMemcpy(
            state.h_ray_hits,
            state.params.ray_primitive_hits,
            state.window * sizeof(unsigned),
            cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
            state.h_ray_intersections,
            state.params.ray_intersections,
            state.window * sizeof(unsigned),
            cudaMemcpyDeviceToHost));
}

void calc_total_hit_intersection_each_window(ScanState &state) {
    int ray_num = OPTIMIZATION == 1 ? state.params.ray_origin_num : state.window;
    if (OPTIMIZATION == 0 && state.launch_ray_num != 0) {
        ray_num = state.launch_ray_num;
    }
    int bvh_node_num = OPTIMIZATION == 2 ? state.params.ray_origin_num : state.window;
    unsigned total_hit = 0, total_intersection_test = 0;
    for (int i = 0; i < ray_num; i++) {
        total_hit += state.h_ray_hits[i];
        total_intersection_test += state.h_ray_intersections[i];
    }
    state.total_hit     += total_hit;
    state.total_is_test += total_intersection_test;
    state.total_is_test_per_ray += 1.0 * total_intersection_test / ray_num;
    state.total_hit_per_ray     += 1.0 * total_hit / ray_num;
    std::cout << "Total_hit: " << total_hit << ", " << "Total_intersection_test: " << total_intersection_test << std::endl;
    std::cout << "BVH_Node: " << bvh_node_num << ", Cast_Ray_Num: " << ray_num << std::endl;
}

void calc_ray_hits(ScanState &state, unsigned *ray_hits) {
    map<unsigned, int> hitNum_rayNum;
    int sum = 0;
    int ray_num = OPTIMIZATION == 1 ? state.params.ray_origin_num : state.window;
    if (OPTIMIZATION == 0 && state.launch_ray_num != 0) {
        ray_num = state.launch_ray_num;
    }
    for (int i = 0; i < ray_num; i++) {
        sum += ray_hits[i];
        if (hitNum_rayNum.count(ray_hits[i])) {
            hitNum_rayNum[ray_hits[i]]++;
        } else {
            hitNum_rayNum[ray_hits[i]] = 1;
        }       
    }

    int min, max, median = -1;
    double avg;
    int tmp_sum = 0;
    min = hitNum_rayNum.begin()->first;
    max = (--hitNum_rayNum.end())->first;
    avg = 1.0 * sum / ray_num;
    printf("hit num: ray num\n");
    for (auto &item: hitNum_rayNum) {
        fprintf(stdout, "%d: %d\n", item.first, item.second);
        tmp_sum += item.second;
        if (median == -1 && tmp_sum >= ray_num / 2) {
            median = item.first;
        }
    }
    printf("min: %d, max: %d, average: %lf, median: %d\n", min, max, avg, median);
}

void check_outlier(int *outlier_list, int outlier_num, double3 *vertices_window, int window_id, int unit_num, ScanState &state) {
    map<int, int> check_outlier_list;
    int check_outlier_num = 0;
    for (int i = 0; i < state.window; i++) {
        int current_neighbor_num = 0;
        for (int j = 0; j < state.window; j++) {
            double3 O = { vertices_window[i].x - vertices_window[j].x, 
                          vertices_window[i].y - vertices_window[j].y,
                          vertices_window[i].z - vertices_window[j].z };
            double sqdist = O.x * O.x + O.y * O.y + O.z * O.z;
            if (sqdist < state.params.R2) {
                current_neighbor_num++;
                if (current_neighbor_num > state.K) {
                    break;
                }
            }
        }
        if (current_neighbor_num <= state.K) {
            check_outlier_list[i] = current_neighbor_num; // * include itself
            check_outlier_num++;
        }
    }

    // cout << "[CHECK] check_outlier_num: " << check_outlier_num << endl;

    // print each outlier index and its neighbor number
    // cout << "check_outlier:check_outlier_neightbor_num" << endl;
    // for (auto it = check_outlier_list.begin(); it != check_outlier_list.end(); it++) {
    //     cout << it->first << ":" << it->second << endl;
    // }

    // check the result
    if (check_outlier_num != outlier_num) {
        cerr << "[Error outlier num in window_id=" << window_id << "], outlier_num=" << outlier_num << ", check_outlier_num=" << check_outlier_num << endl;
        exit(1);
    }
    // for (int i = 0; i < outlier_num; i++) {
    //     int outlier_id = (state.h_outlier_list[i] + (unit_num - window_id % unit_num) * state.slide) % state.window;
    //     if (!check_outlier_list.count(outlier_id)) { // check idx
    //         cerr << "[Error outlier idx in window_id=" << window_id << "], outlier_idx=" << outlier_list[i]
    //              << ", outlier_neighbor_num=" << state.h_outlier_neightbor_num[i] << endl;
    //         exit(1);
    //     }
    // }
}

void cleanup(ScanState &state) {
    // free host memory
    vertices -= state.start_copy_pos * state.slide;
    free(vertices);
#if OPTIMIZATION == 1 || OPTIMIZATION == 2
    free(state.h_current_window);
    free(state.h_ray_origin_list);
#endif
    
    free(state.h_outlier_list);

#if OPTIMIZATION == 2
    free(state.h_ray_origin_idx);
    free(state.h_outlier_neightbor_num);
#endif

    // free device memory
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.sbt.raygenRecord)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.sbt.missRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.sbt.hitgroupRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.d_gas_output_buffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.d_temp_buffer_gas)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.d_aabb_ptr)));

    OPTIX_CHECK(optixPipelineDestroy(state.pipeline));
    OPTIX_CHECK(optixProgramGroupDestroy(state.hitgroup_prog_group));
    OPTIX_CHECK(optixProgramGroupDestroy(state.miss_prog_group));
    OPTIX_CHECK(optixProgramGroupDestroy(state.raygen_prog_group));
    OPTIX_CHECK(optixModuleDestroy(state.module));

    OPTIX_CHECK(optixDeviceContextDestroy(state.context));

    CUDA_CHECK(cudaFree(state.params.outlier_num));
#if OPTIMIZATION == 0 || OPTIMIZATION == 1
    CUDA_CHECK(cudaFree(state.params.outlier_list));
#endif
    CUDA_CHECK(cudaFree(state.params.points));
#if OPTIMIZATION == 2
    CUDA_CHECK(cudaFree(state.params.outlier_neighbor_num));
#endif

#if DEBUG_INFO == 1
    CUDA_CHECK(cudaFree(state.params.ray_primitive_hits));
    CUDA_CHECK(cudaFree(state.params.ray_intersections));
    free(state.h_ray_hits);
    free(state.h_ray_intersections);
#endif

    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.d_params)));
}

void log_common_info(ScanState &state) {
    std::cout << "Dimension: " << DIMENSION << std::endl;
    std::cout << "Compaction: " << COMPACTION << std::endl;
    std::cout << "Update GAS type: " << ((UPDATE_GAS_TYPE == 0) ? "Update" : "Rebuild") << std::endl;
    std::cout << "data num: " << data_num << std::endl;
    std::cout << "R: " << state.R << ", R2: " << state.params.R2 << std::endl;
    std::cout << "K: " << state.K << std::endl;
    std::cout << "Window size: " << state.window << std::endl;
    std::cout << "Slide size: " << state.slide << std::endl;
    std::cout << "Input file: " << state.infile << std::endl;
    std::cout << "Optimization: " << OPTIMIZATION << std::endl;
}

int get_cell_id(ScanState &state, int i, bool add) {
    int id = 0;
#if DIMENSION == 1
    id = (vertices[i].x - state.min_value[0]) / state.cell_length;
#elif DIMENSION == 3
    int dim_id_x = (vertices[i].x - state.min_value[0]) / state.cell_length;
    int dim_id_y = (vertices[i].y - state.min_value[1]) / state.cell_length;
    int dim_id_z = (vertices[i].z - state.min_value[2]) / state.cell_length;
    id = dim_id_x * state.cell_count[1] * state.cell_count[2] + dim_id_y * state.cell_count[2] + dim_id_z;
#endif
    return id;
}

void initialize_cell(ScanState &state) {
    state.cell_length = state.R / sqrt(DIMENSION);
    for (int i = 0; i < DIMENSION; i++) {
        state.cell_count[i] = int((state.max_value[i] - state.min_value[i] + state.cell_length) / state.cell_length);
    }
}

void prepare_c_non_points_queue(ScanState &state, int window_left, int window_right, int update_pos) {
    // expired points
    for (int i = window_left; i < window_left + state.slide; i++) {
        int cell_id = get_cell_id(state, i, false);
        FixQueue &fq = state.cell_queue[cell_id];
        fq.num--;
        if (fq.num == 0) {
            state.cell_queue.erase(cell_id);
        }
    }
    // new points
    for (int i = window_right; i < window_right + state.slide; i++) {
        int cell_id = get_cell_id(state, i, true);
        state.cell_queue[cell_id].enqueue(update_pos * state.slide + i - window_right); // 记录 id
    }

    // classify cell
    state.undetermined_cell_list.clear();
    int undetermined_point_num = 0;
    int undetermined_cell_num  = 0;
    for (auto it = state.cell_queue.begin(); it != state.cell_queue.end(); it++) {
        if (it->second.num <= state.K) {
            state.undetermined_cell_list.push_back(it->first);
            undetermined_cell_num++;
            undetermined_point_num += it->second.num;
        }
    }

    // set state.h_ray_origin_list
    int device_pos = 0;
    for (int i = 0; i < undetermined_cell_num; i++) {
        FixQueue &q = state.cell_queue[state.undetermined_cell_list[i]];
#if OPTIMIZATION == 1
        q.copy(state.h_ray_origin_list + device_pos, state.h_current_window);
#elif OPTIMIZATION == 2
        q.copy(state.h_ray_origin_list + device_pos, state.h_ray_origin_idx + device_pos, state.h_current_window);
#endif
        device_pos += q.num;
    }

    state.params.ray_origin_num = undetermined_point_num;
    state.total_cast_rays      += undetermined_point_num;
    // transfer state.h_ray_origin_list to device
    timer.startTimer(&timer.copy_filtered_points_h2d);
    CUDA_CHECK(cudaMemcpy(
            state.params.ray_origin_list,
            state.h_ray_origin_list,
            state.params.ray_origin_num * sizeof(double3),
            cudaMemcpyHostToDevice));
    timer.stopTimer(&timer.copy_filtered_points_h2d);
}

void detect_outlier(ScanState &state, bool warmup) {
    CUDA_CHECK(cudaMemcpy(
        state.params.points,
        vertices,
        state.window * sizeof(double3),
        cudaMemcpyHostToDevice));
#if OPTIMIZATION == 0 || OPTIMIZATION == 1
    kGenAABB(state.params.points, state.R, state.window, reinterpret_cast<OptixAabb*>(state.d_aabb_ptr)); 
#endif
#if OPTIMIZATION == 1 || OPTIMIZATION == 2
    memcpy(state.h_current_window, vertices, state.window * sizeof(double3));
    for (int i = 0; i < state.window; i++) {
        int cell_id = get_cell_id(state, i, true);
        state.cell_queue[cell_id].enqueue(i);
    }
#endif

    int remaining_data_num  = data_num - state.window;
    int unit_num            = state.window / state.slide;
    int update_pos          = 0;
    int slide_num           = 0;
    int window_left         = 0;
    int window_right        = state.window;
    state.new_slide         = vertices + state.window;

    // * start sliding
    while (remaining_data_num >= state.slide && slide_num < 10000) {
        timer.startTimer(&timer.total);
        CUDA_CHECK(cudaMemset(state.params.outlier_num, 0, sizeof(int)));
#if OPTIMIZATION == 2
        CUDA_CHECK(cudaMemset(state.params.outlier_neighbor_num, 0, state.window * sizeof(int)));
#endif
        timer.startTimer(&timer.copy_new_points_h2d);
        // transfer new slide to the device
        CUDA_CHECK(cudaMemcpy(
            state.params.points + update_pos * state.slide,
            state.new_slide,
            state.slide * sizeof(double3),
            cudaMemcpyHostToDevice));
        timer.stopTimer(&timer.copy_new_points_h2d);

#if OPTIMIZATION == 1 || OPTIMIZATION == 2
        // prepare current window in host
        memcpy(state.h_current_window + update_pos * state.slide,
            state.new_slide,
            state.slide * sizeof(double3));
        timer.startTimer(&timer.prepare_cell);
        prepare_c_non_points_queue(state, window_left, window_right, update_pos);
        timer.stopTimer(&timer.prepare_cell);
#endif

        timer.startTimer(&timer.build_bvh);
#if UPDATE_GAS_TYPE == 0
        update_gas(state, update_pos);
#else
        rebuild_gas(state, update_pos);
#endif
        CUDA_SYNC_CHECK();
        timer.stopTimer(&timer.build_bvh);

        timer.startTimer(&timer.detect_outlier);
        launch(state);
        timer.stopTimer(&timer.detect_outlier);
        
        // * D2H
        timer.startTimer(&timer.copy_outlier_d2h);
        int outlier_num = 0;
#if OPTIMIZATION == 0 || OPTIMIZATION == 1
        CUDA_CHECK(cudaMemcpy(&outlier_num, state.params.outlier_num, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(state.h_outlier_list, state.params.outlier_list, outlier_num * sizeof(int), cudaMemcpyDeviceToHost));
#else
        CUDA_CHECK(cudaMemcpy(
            state.h_outlier_neightbor_num,
            state.params.outlier_neighbor_num,
            state.params.ray_origin_num * sizeof(int),
            cudaMemcpyDeviceToHost));
        for (int i = 0; i < state.params.ray_origin_num; i++) {
            if (state.h_outlier_neightbor_num[i] <= state.K) {
                state.h_outlier_list[outlier_num++] = state.h_ray_origin_idx[i];
            }
        }
#endif
        timer.stopTimer(&timer.copy_outlier_d2h);

        slide_num++;
        remaining_data_num  -= state.slide;
        state.new_slide     += state.slide;
        update_pos           = (update_pos + 1) % unit_num;
        window_left         += state.slide;
        window_right        += state.slide;

        timer.stopTimer(&timer.total);

#if DEBUG_INFO == 1
        if (!warmup) {
            cout << "At window " << slide_num << ", # outliers: " << outlier_num << endl;
            result_d2h(state, outlier_num, slide_num, unit_num);
            calc_total_hit_intersection_each_window(state);
            // calc_ray_hits(state, state.h_ray_hits);
            // check_outlier(state.h_outlier_list, outlier_num, vertices + slide_num * state.slide, slide_num, unit_num, state);
        }
#endif
    }

    if (warmup) {
        state.cell_queue.clear();
    }
}

int main(int argc, char *argv[])
{
    ScanState state;
    parse_args(state, argc, argv);
    size_t start_gpu_memory;
    start_gpu_mem(&start_gpu_memory);

    read_data(state.infile, state);
    vertices += state.start_copy_pos * state.slide;

    initialize_optix(state);
    
    size_t optix_context_used;
    stop_gpu_mem(&start_gpu_memory, &optix_context_used);
    std::cout << "[Mem] Optix context used(MB): " << 1.0 * optix_context_used / (1 << 20) << std::endl;
    start_gpu_mem(&start_gpu_memory);

    data_h2d(state);
    make_gas(state);                    // Acceleration handling
    make_module(state);
    make_program_groups(state);
    make_pipeline(state);               // Link pipeline; Occupy most cpu memory
    make_sbt(state);
    
    size_t init_cpu_mem = get_cpu_memory_usage();
    initialize_params(state);
    log_common_info(state);

#if OPTIMIZATION == 1 || OPTIMIZATION == 2
    initialize_cell(state);
#endif
    for (int i = 0; i < 10; i++) {
        detect_outlier(state, true);    // warmup
    }
    timer.clearNew();
    detect_outlier(state, false);       // timing

    int slide_num = (data_num - state.window) / state.slide;
    if (slide_num == 0) slide_num = 1;
    timer.average(slide_num);
    timer.showTimeNew();
    
    size_t used_cpu_mem = get_cpu_memory_usage() - init_cpu_mem;
    std::cout << "[Mem] Host cpu memery used(MB): " << 1.0 * used_cpu_mem / (1 << 10) << std::endl;
    size_t rtod_used_gpu_memory;
    stop_gpu_mem(&start_gpu_memory, &rtod_used_gpu_memory);
    std::cout << "[Mem] Device memory used for data(MB): " << 1.0 * rtod_used_gpu_memory / (1 << 20) << std::endl;

#if OPTIMIZATION == 1
    std::cout << "Ray / slide: " << state.total_cast_rays / slide_num << std::endl;
#elif OPTIMIZATION == 2
    std::cout << "BVH Node / slide: " << state.total_cast_rays / slide_num << std::endl;
#endif

#if DEBUG_INFO == 1
    std::cout << "Hit / slide: " << state.total_hit / slide_num << std::endl;
    std::cout << "Intersection test / slide: " << state.total_is_test / slide_num << std::endl;
    std::cout << "Intersection test per ray on average: " << state.total_is_test_per_ray / slide_num << std::endl;
    std::cout << "Hit per ray on average: " << state.total_hit_per_ray / slide_num << endl;
#endif
    cleanup(state);
    return 0;
}
