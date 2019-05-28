#pragma once
#include <float.h>
#include <vector_types.h>
#include "simplex.cuh"
#include "vector_operations.cuh"
#include "BoxCollider.h"
#define CCD_INIT(ccd) \
    do { \
        (ccd)->max_iterations = (unsigned long)-1; \
        (ccd)->epa_tolerance = 0.0001f; \
        (ccd)->mpr_tolerance = 0.0001f; \
        (ccd)->dist_tolerance = 1E-6f; \
    } while(0)

#define ZERO 0.0f
#define ONE 1.0f
#define COL_EPS FLT_EPSILON


struct GJKData{
	unsigned long max_iterations; //!< Maximal number of iterations
	float epa_tolerance;
	float mpr_tolerance; //!< Boundary tolerance for MPR algorithm
	float dist_tolerance;
};

typedef struct GJKData GJKData_t;


__device__ void firstDir(const void *obj1, const void *obj2, float3 *dir);
__device__ void supportBox(const void *_obj, const float3 *_dir, float3 *v);
__device__ int GJKIntersect(const void *obj1, const void *obj2, const GJKData_t *data);
__device__ static int GJK(const void *obj1, const void *obj2,
				const GJKData_t *data, simplex_t *simplex);
__device__ void SupportCalc(const void *obj1, const void *obj2, const float3 *_dir, support_t *supp);
__device__ static int doSimplex(simplex_t *simplex, float3 *dir);
__device__ static int doSimplex2(simplex_t *simplex, float3 *dir);
__device__ static int doSimplex3(simplex_t *simplex, float3 *dir);
__device__ static int doSimplex4(simplex_t *simplex, float3 *dir);
__device__ float Vec3PointTriDist2(const float3 *P, const float3 *x0, const float3 *B, const float3 *C, float3 *witness);
__device__ float PointSegmentDist(const float3 *P, const float3 *x0, const float3 *b, float3 *witness);
__device__ void quatRotVec(float3 *v, const float4 *q);
__device__ int quatInvert2(float4 *dest, const float4 *src);
__device__ int quatInvert(float4 *q);
__device__ int Signum(float val);
__device__ void tripleCross(const float3 *a, const float3 *b,
				const float3 *c, float3 *d);
__device__ int IsZERO(float val);
__device__ int floatEq(float a, float b);
__device__ float absolute(float val);