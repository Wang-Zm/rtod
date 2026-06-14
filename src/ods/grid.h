#ifndef GRID_H
#define GRID_H

#include "state.h"

void initialize_cell(ScanState &state);
int  get_cell_id(ScanState &state, int i, bool add);
void prepare_c_non_points_queue(ScanState &state, int window_left, int window_right, int update_pos);

#endif