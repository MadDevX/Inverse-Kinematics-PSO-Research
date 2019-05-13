#pragma once

__device__ float3 operator+(float3 f1, float3 f2);

__device__ float3 operator-(float3 f1, float3 f2);

__device__ float3 operator*(float3 f1, float a);

__device__ float4 operator+(float4 f1, float4 f2);

__device__ float4 operator-(float4 f1, float4 f2);

__device__ float4 operator*(float4 f1, float a);

__device__ float magnitudeSqr(const float3 vector);

__device__ float magnitudeSqr(const float4 vector);

__device__ void float3Scale(float3* v, float a);

__device__ void float3Sub(float3* V, const  float3* v1, const float3* v2);

__device__ void float3Add(float3* V, const float3* v1, const float3* v2);

__device__ float float3Dot(const float3* v1, const  float3* v2);

__device__ void float3Cross(float3* d, const  float3* a, const float3* b);

__device__ void float4Copy(float4* V, const  float4* v1);

__device__ void float3Copy(float3* V, const float3* v1);

__device__ float float3Len(float3 *v);

__device__ bool float3Eq(const float3 *v1, const float3 *v2);

__device__ float float3Dist(const  float3 * v1, const float3 * v2);