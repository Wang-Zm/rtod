#ifndef BVH_H
#define BVH_H

#include "state.h"

extern "C" void kGenAABB(double3 *points, double radius, unsigned int numPrims, OptixAabb *d_aabb);

void make_gas(ScanState &state);
void update_gas(ScanState &state, int update_pos);
void rebuild_gas(ScanState &state, int update_pos);

#endif