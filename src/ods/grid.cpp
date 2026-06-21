#include "grid.h"
#include "timer.h"

#include <cuda_runtime.h>

#include <cmath>

using namespace std;

extern double3*  vertices;
extern Timer     timer;

int get_cell_id(ScanState &state, int i) {
    int id = 0;
    if constexpr (DIMENSION == 1) {
        id = (vertices[i].x - state.min_value[0]) / state.cell_length;
    } else if constexpr (DIMENSION == 3) {
        int dim_id_x = (vertices[i].x - state.min_value[0]) / state.cell_length;
        int dim_id_y = (vertices[i].y - state.min_value[1]) / state.cell_length;
        int dim_id_z = (vertices[i].z - state.min_value[2]) / state.cell_length;
        id = dim_id_x * state.cell_count[1] * state.cell_count[2] + dim_id_y * state.cell_count[2] + dim_id_z;
    }
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
        int cell_id = get_cell_id(state, i);
        FixQueue &fq = state.cell_queue[cell_id];
        fq.num--;
        if (fq.num == 0) {
            state.cell_queue.erase(cell_id);
        }
    }
    // new points
    for (int i = window_right; i < window_right + state.slide; i++) {
        int cell_id = get_cell_id(state, i);
        state.cell_queue[cell_id].enqueue(update_pos * state.slide + (i - window_right)); // record id
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
        if constexpr (OPTIMIZATION == 1) {
            q.copy(state.h_ray_origin_list + device_pos, state.h_current_window);
        } else if constexpr (OPTIMIZATION == 2) {
            q.copy(state.h_ray_origin_list + device_pos, state.h_ray_origin_idx + device_pos, state.h_current_window);
        }
        device_pos += q.num;
    }

    state.params.ray_origin_num = undetermined_point_num;
    state.total_cast_rays      += undetermined_point_num;
    // transfer state.h_ray_origin_list to device
    timer.start(TIMER_copy_filtered_points_h2d);
    CUDA_CHECK(cudaMemcpy(
            state.params.ray_origin_list,
            state.h_ray_origin_list,
            state.params.ray_origin_num * sizeof(double3),
            cudaMemcpyHostToDevice));
    timer.stop(TIMER_copy_filtered_points_h2d);
}