#include <cassert>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

#include <sutil/Exception.h>
#include <sutil/sutil.h>

#include "pipeline.h"
#include "bvh.h"
#include "grid.h"
#include "verify.h"
#include "state.h"
#include "timer.h"

#include <cstring>
#include <iostream>
#include <fstream>
#include <string>

using namespace std;

int                      data_num = DATA_N;
double3*                 vertices;
Timer                    timer;


// ── utility ──────────────────────────────────────────────────────────────

void printUsageAndExit(const char* argv0) {
    std::cerr << "Usage  : " << argv0 << " [options]\n";
    std::cerr << "Options: --file | -f <filename>      Specify file for data input\n";
    std::cerr << "         --help | -h                 Print this usage message\n";
    std::cerr << "         --n <int>                   Set data num; defaults to 1e8\n";
    std::cerr << "         --window <int>              Window size\n";
    std::cerr << "         --slide <int>               Slide size\n";
    std::cerr << "         --R <double>                Distance threshold\n";
    std::cerr << "         --K <int>                   Neighbor threshold\n";
    std::cerr << "         --start_copy_pos <int>      Starting position offset\n";
    std::cerr << "         --launch_ray_num <int>      Override number of rays to launch\n";
    exit(1);
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


// ── data I/O ─────────────────────────────────────────────────────────────

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


// ── argument parsing ─────────────────────────────────────────────────────

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


// ── GPU memory & params setup ────────────────────────────────────────────

void data_h2d(ScanState &state) {
    size_t start, used;
    start_gpu_mem(&start);
    CUDA_CHECK(cudaMalloc(&state.params.points, state.window * sizeof(double3)));
    stop_gpu_mem(&start, &used);
    std::cout << "[Mem] data_h2d: " << 1.0 * used / (1 << 20) << std::endl;
}

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
    state.params.window_size = state.window;
    state.params.handle = state.gas_handle;
    state.params.tmin   = 0.0f;
    state.params.tmax   = FLT_MIN;

    size_t used;
    stop_gpu_mem(&start, &used);
    std::cout << "[Mem] initialize_params: " << 1.0 * used / (1 << 20) << std::endl;
}


// ── launch ───────────────────────────────────────────────────────────────

void launch(ScanState &state) {
    assert(state.params.ray_origin_num <= state.window);
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


// ── cleanup ──────────────────────────────────────────────────────────────

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


// ── runtime validation ───────────────────────────────────────────────────

void validate_params(ScanState &state) {
    // These are logic errors — if they fire, fix the CLI args or CMake flags.
    if (state.window <= 0) {
        cerr << "[FATAL] window must be > 0, got " << state.window << endl;
        exit(1);
    }
    if (state.slide <= 0) {
        cerr << "[FATAL] slide must be > 0, got " << state.slide << endl;
        exit(1);
    }
    if (state.R <= 0) {
        cerr << "[FATAL] R must be > 0, got " << state.R << endl;
        exit(1);
    }
    if (state.K <= 0) {
        cerr << "[FATAL] K must be > 0, got " << state.K << endl;
        exit(1);
    }
    if (data_num < state.window) {
        cerr << "[FATAL] data_num (" << data_num << ") must be >= window (" << state.window << ")" << endl;
        exit(1);
    }
    if (state.window % state.slide != 0) {
        cerr << "[FATAL] window (" << state.window << ") must be divisible by slide ("
             << state.slide << "), otherwise ring-buffer indexing breaks." << endl;
        exit(1);
    }
    if (state.K > MK) {
        cerr << "[FATAL] K (" << state.K << ") must be <= MK (" << MK
             << "). FixQueue internally uses arr[MK], so overflow would occur. "
                "Rebuild with -D MK=" << state.K << endl;
        exit(1);
    }
    if (state.launch_ray_num > state.window) {
        cerr << "[FATAL] launch_ray_num (" << state.launch_ray_num
             << ") must be <= window (" << state.window << ")" << endl;
        exit(1);
    }
    std::cout << "[Invariant] All runtime parameter checks passed." << std::endl;
}


// ── logging ──────────────────────────────────────────────────────────────

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


// ── core sliding-window loop ─────────────────────────────────────────────

extern "C" void kGenAABB(double3 *points, double radius, unsigned int numPrims, OptixAabb *d_aabb);

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
        int cell_id = get_cell_id(state, i);
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
        assert(state.params.ray_origin_num <= state.window);
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
        assert(outlier_num <= state.window);
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


// ── main ─────────────────────────────────────────────────────────────────

int main(int argc, char *argv[])
{
    ScanState state;
    parse_args(state, argc, argv);
    validate_params(state);
    size_t start_gpu_memory;
    start_gpu_mem(&start_gpu_memory);

    read_data(state.infile, state);
    vertices += state.start_copy_pos * state.slide;

    initialize_optix(state);

    size_t optix_context_used;
    stop_gpu_mem(&start_gpu_memory, &optix_context_used);
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
    std::cout << "[Mem] Host cpu memory used(MB): " << 1.0 * used_cpu_mem / (1 << 10) << std::endl;
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