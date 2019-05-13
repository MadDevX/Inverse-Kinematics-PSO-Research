#pragma once
#include "support.cuh"

struct simplex {
	support_t ps[4];
	int last; //!< index of last added point
};
typedef struct simplex simplex_t;

__device__ void SimplexInit(simplex_t *s);
__device__ int SimplexSize(const simplex_t *s);
__device__ const support_t *SimplexLast(const simplex_t *s);
__device__ const support_t *SimplexPoint(const simplex_t *s, int idx);
__device__ support_t *SimplexPointW(simplex_t *s, int idx);

__device__ void SimplexAdd(simplex_t *s, const support_t *v);
__device__ void SimplexSet(simplex_t *s, size_t pos, const support_t *a);
__device__ void SimplexSetSize(simplex_t *s, int size);
__device__ void SimplexSwap(simplex_t *s, size_t pos1, size_t pos2);

