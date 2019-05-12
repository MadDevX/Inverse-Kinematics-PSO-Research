#pragma once
#include "cuda_runtime.h"


__device__ float3 operator+(float3 f1, float3 f2)
{

	return make_float3(f1.x + f2.x, f1.y + f2.y, f1.z + f2.z);
}

__device__ float3 operator-(float3 f1, float3 f2)
{
	return make_float3(f1.x - f2.x, f1.y - f2.y, f1.z - f2.z);
}

__device__ float3 operator*(float3 f1, float a)
{
	return make_float3(f1.x *a, f1.y *a, f1.z *a);
}

__device__ float4 operator+(float4 f1, float4 f2)
{
	return make_float4(f1.x + f2.x, f1.y + f2.y, f1.z + f2.z, f1.w + f2.w);
}

__device__ float4 operator-(float4 f1, float4 f2)
{
	return make_float4(f1.x - f2.x, f1.y - f2.y, f1.z - f2.z, f1.w - f2.w);
}

__device__ float4 operator*(float4 f1, float a)
{
	return make_float4(f1.x *a, f1.y *a, f1.z *a, f1.w *a);
}

__device__ float magnitudeSqr(const float3 vector)
{
	return (vector.x * vector.x) + (vector.y * vector.y) + (vector.z * vector.z);
}

__device__ float magnitudeSqr(const float4 vector)
{
	return (vector.x * vector.x) + (vector.y * vector.y) + (vector.z * vector.z) + (vector.w * vector.w);
}



__device__ void float3Scale(float3* v, float a)
{
	(*v) = (*v) * a;
}

__device__ void float3Sub(float3* V, const  float3* v1, const float3* v2)
{
	(*V) = (*v1) - (*v2);
}

__device__ void float3Add(float3* V, const float3* v1, const float3* v2)
{
	(*V) = (*v1) + (*v2);
}

__device__ float float3Dot(const float3* v1, const  float3* v2)
{
	float dot = 0.0f;
	dot =  v1->x * v2->x;
	dot += v1->y * v2->y;
	dot += v1->z * v2->z;
	return dot;
}

__device__ void float3Cross(float3* d, const  float3* a, const float3* b)
{
	d->x = (a->y * b->z) - (a->z * b->y);
	d->y = (a->z * b->x) - (a->x * b->z);
	d->z = (a->x * b->y) - (a->y * b->x);

}


__device__ void float4Copy(float4* V, const  float4* v1)
{
	V->x = v1->x;
	V->y = v1->y;
	V->z = v1->z;
	V->w = v1->w;
}

__device__ void float3Copy(float3* V, const float3* v1)
{
	V->x = v1->x;
	V->y = v1->y;
	V->z = v1->z;
}

__device__ float float3Len(float3 *v)
{
	return float3Dot(v, v);
}


__device__ bool float3Eq(const float3 *v1, const float3 *v2)
{
	return  (v1->x == v2->x) &&
		(v1->y == v2->y) &&
		(v1->z == v2->z);
}

__device__ float float3Dist(const  float3 * v1, const float3 * v2)
{
	float3 res = (*v1) - (*v2);
	return float3Len(&(res));
}