#include <optix.h>
#include <sutil/vec_math.h>
#include "optixScan.h"

__global__ void kGenAABB_t (
      double3* points,
      double radius,
      unsigned int N,
      OptixAabb* aabb
)
{
  unsigned int particleIndex = blockIdx.x * blockDim.x + threadIdx.x;
  if (particleIndex >= N) return;

  double3 center = points[particleIndex];

  // float3 m_min = center - radius;
  // float3 m_max = center + radius;
  float3 m_min;
  float3 m_max;
  m_min.x = center.x - radius;
  m_min.y = center.y - radius;
  m_min.z = center.z - radius;
  m_max.x = center.x + radius;
  m_max.y = center.y + radius;
  m_max.z = center.z + radius;

  aabb[particleIndex] =
  {
    m_min.x, m_min.y, m_min.z,
    m_max.x, m_max.y, m_max.z
  };
}

extern "C" void kGenAABB(double3* points, double width, unsigned int numPrims, OptixAabb* d_aabb) {
  unsigned int threadsPerBlock = 64;
  unsigned int numOfBlocks = numPrims / threadsPerBlock + 1;

  kGenAABB_t <<<numOfBlocks, threadsPerBlock>>> (
    points,
    width,
    numPrims,
    d_aabb
    );
}
