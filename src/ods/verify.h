#ifndef VERIFY_H
#define VERIFY_H

#include "state.h"

void result_d2h(ScanState &state, int outlier_num, int window_id, int unit_num);
void calc_total_hit_intersection_each_window(ScanState &state);
void calc_ray_hits(ScanState &state, unsigned *ray_hits);
void check_outlier(int *outlier_list, int outlier_num, double3 *vertices_window, int window_id, int unit_num, ScanState &state);

#endif