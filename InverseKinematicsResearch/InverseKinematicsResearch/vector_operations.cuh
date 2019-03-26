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

__device__ float4 operator+(float4 f1, float4 f2)
{
	return make_float4(f1.x + f2.x, f1.y + f2.y, f1.z + f2.z, f1.w + f2.w);
}

__device__ float4 operator-(float4 f1, float4 f2)
{
	return make_float4(f1.x - f2.x, f1.y - f2.y, f1.z - f2.z, f1.w - f2.w);
}

__device__ float magnitudeSqr(float3 vector)
{
	return (vector.x * vector.x) + (vector.y * vector.y) + (vector.z * vector.z);
}

__device__ float magnitudeSqr(float4 vector)
{
	return (vector.x * vector.x) + (vector.y * vector.y) + (vector.z * vector.z) + (vector.w * vector.w);
}
