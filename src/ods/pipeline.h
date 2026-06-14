#ifndef PIPELINE_H
#define PIPELINE_H

#include <optix.h>
#include "state.h"

template <typename T>
struct SbtRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData>   RayGenSbtRecord;
typedef SbtRecord<MissData>     MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;

void initialize_optix(ScanState &state);
void make_module(ScanState &state);
void make_program_groups(ScanState &state);
void make_pipeline(ScanState &state);
void make_sbt(ScanState &state);

#endif