#pragma once
#include "cuda_runtime.h"
#include <math.h>

__device__ float4 eulerToQuaternion(float3 eulerAngles)
{
	float cx = cosf(eulerAngles.x * 0.5f);
	float cy = cosf(eulerAngles.y * 0.5f);
	float cz = cosf(eulerAngles.z * 0.5f);
	float sx = sinf(eulerAngles.x * 0.5f);
	float sy = sinf(eulerAngles.y * 0.5f);
	float sz = sinf(eulerAngles.z * 0.5f);

	float4 q;
	q.w = cx * cy * cz + sx * sy * sz;
	q.x = sx * cy * cz - cx * sy * sz;
	q.y = cx * sy * cz + sx * cy * sz;
	q.z = cx * cy * sz - sx * sy * cz;
	return q;
}

__device__ float3 quaternionToEuler(float4 q)
{
	float3 angles;
	float dividend, divisor;

	dividend = 2.0f * (q.w*q.x + q.y*q.z);
	divisor = 1.0f - 2.0f*(q.x*q.x + q.y*q.y);
	angles.x = atan2f(dividend, divisor);

	angles.y = asinf(2.0f * (q.w*q.y - q.z*q.x));

	dividend = 2.0f * (q.w*q.z + q.y*q.x);
	divisor = 1.0f - 2.0f*(q.z*q.z + q.y*q.y);
	angles.z = atan2f(dividend, divisor);

	return angles;
}