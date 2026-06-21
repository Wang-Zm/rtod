#include "verify.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
#include <map>

using namespace std;

void result_d2h(ScanState &state, int /*outlier_num*/, int /*window_id*/, int /*unit_num*/) {
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

    if (hitNum_rayNum.empty()) return;  // guard: no rays were cast (e.g. ray_origin_num == 0)

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

    // check the result
    if (check_outlier_num != outlier_num) {
        cerr << "[Error outlier num in window_id=" << window_id << "], outlier_num=" << outlier_num << ", check_outlier_num=" << check_outlier_num << endl;
        cout << "[Error outlier num in window_id=" << window_id << "], outlier_num=" << outlier_num << ", check_outlier_num=" << check_outlier_num << endl;
        cout << "RTOD outliers:" << endl;
        sort(outlier_list, outlier_list + outlier_num);
        for (int i = 0; i < outlier_num; i++) {
            cout << i << ": " << outlier_list[i] << endl;
        }
        cout << "Naive outliers:" << endl;
        int onum = 0;
        for (auto outlier: check_outlier_list) {
            cout << onum << ": " << outlier.first << endl;
            onum++;
        }
        cout << "BVH node index:" << endl;
        sort(state.h_ray_origin_idx, state.h_ray_origin_idx + state.params.ray_origin_num);
        for (int i = 0; i < state.params.ray_origin_num; i++) {
            cout << i << ": " << state.h_ray_origin_idx[i] << endl;
            if (i > 0 && state.h_ray_origin_idx[i] == state.h_ray_origin_idx[i - 1]) {
                cout << "repeated idx: " << state.h_ray_origin_idx[i] << endl;
            }
        }
        for (int i = 0; i < outlier_num; i++) {
            int outlier_id = outlier_list[i];
            if (!check_outlier_list.count(outlier_id)) {
                cerr << "[Error outlier idx in window_id=" << window_id << "], outlier_idx=" << outlier_list[i]
                    << ", outlier_neighbor_num=" << state.h_outlier_neightbor_num[i] << endl;
                exit(1);
            }
        }
        exit(1);
    }
}