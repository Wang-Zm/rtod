#ifndef STATE_H
#define STATE_H

#include <float.h>
#include <vector_types.h>
#include <optix_types.h>
#include <unordered_set>
#include <unordered_map>
#include <list>
#include <deque>
#include "optixScan.h"

using namespace std;

class FixQueue {
public:
    int         arr[MK];
    int         start;
    int         num;

    FixQueue() {
        start = 0;
        num = 0;
    }

    void enqueue(int val) {
        arr[start] = val;
        if ((++start) == MK) {
            start = 0;
        }
        num++;
    }

    void copy(double3* dsc, int* dsc_idx, double3* h_current_window) {
        int _start = (start - 1 + MK) % MK;
        int _num   = num;
        while (_num > 0) {
            *dsc = h_current_window[arr[_start]];
            *dsc_idx = arr[_start];
            dsc++;
            dsc_idx++;
            _start = (_start - 1 + MK) % MK;
            _num--;
        }
    }

    void copy(double3* dsc, double3* h_current_window) {
        int _start = (start - 1 + MK) % MK;
        int _num   = num;
        while (_num > 0) {
            *dsc = h_current_window[arr[_start]];
            dsc++;
            _start = (_start - 1 + MK) % MK;
            _num--;
        }
    }
};

struct ScanState
{
    Params                          params;
    CUdeviceptr                     d_params;
    OptixDeviceContext              context                   = nullptr;
    OptixTraversableHandle          gas_handle;
    CUdeviceptr                     d_gas_output_buffer       = 0;
    OptixBuildInput                 vertex_input              = {};
    CUdeviceptr                     d_temp_buffer_gas         = 0;
    OptixAccelBufferSizes           gas_buffer_sizes;
    const uint32_t                  vertex_input_flags[1]     = {OPTIX_GEOMETRY_FLAG_NONE};

    OptixModule                     module                    = nullptr;

    OptixProgramGroup               raygen_prog_group         = nullptr;
    OptixProgramGroup               miss_prog_group           = nullptr;
    OptixProgramGroup               hitgroup_prog_group       = nullptr;

    OptixPipeline                   pipeline                  = nullptr;
    OptixPipelineCompileOptions     pipeline_compile_options  = {};

    OptixShaderBindingTable         sbt                       = {}; 

    std::string                     infile;
    int                             window;
    int                             slide;
    double                          R;
    int                             K;
    double3*                        new_slide;
    CUdeviceptr                     d_aabb_ptr                = 0;

    unordered_map<int, FixQueue>    cell_queue;
    double                          cell_length;
    int                             cell_count[DIMENSION];
    vector<int>                     undetermined_cell_list;

    double                          max_value[DIMENSION];
    double                          min_value[DIMENSION];
    double3*                        h_current_window;
    double3*                        h_ray_origin_list;

    int*                            h_ray_origin_idx; // h_ray_origin_list 中每个点在当前 window 中的 idx
    int*                            h_outlier_neightbor_num;

    int*                            h_outlier_list;

    
    unsigned*                       h_ray_hits;
    unsigned*                       h_ray_intersections;

    long                            total_cast_rays           = 0;
    double                          total_is_test_per_ray     = 0;
    double                          total_hit_per_ray         = 0;
    long                            total_hit                 = 0;
    long                            total_is_test             = 0;

    int                             start_copy_pos            = 0;
    int                             launch_ray_num            = 0;
};


#endif