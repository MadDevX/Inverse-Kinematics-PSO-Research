#pragma once
#include "cuda_runtime.h"
#include <math.h>

struct Matrix
{
	float cells[16];
};

__device__ Matrix createMatrix(float f)
{
	Matrix mat;
	for (int i = 0; i < 16; i++)
	{
		mat.cells[i] = 0.0f;
	}

	for (int i = 0; i < 4; i++)
	{
		mat.cells[i + 4 * i] = f;
	}
	return mat;
}

__device__ Matrix multiplyMatrices(Matrix left, Matrix right)
{
	Matrix result = createMatrix(0.0f);

	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			float sum = 0.0f;
			for (int x = 0; x < 4; x++)
			{
				sum += left.cells[x + j * 4] * right.cells[x * 4 + i];
			}
			result.cells[i + j * 4] = sum;
		}
	}

	return result;
}

__device__ float4 multiplyMatByVec(Matrix left, float4 vector)
{
	float4 result;
	result.x = left.cells[0] * vector.x + left.cells[1] * vector.y + left.cells[2] * vector.z + left.cells[3] * vector.w;
	result.y = left.cells[4] * vector.x + left.cells[5] * vector.y + left.cells[6] * vector.z + left.cells[7] * vector.w;
	result.z = left.cells[8] * vector.x + left.cells[9] * vector.y + left.cells[10] * vector.z + left.cells[11] * vector.w;
	result.w = left.cells[12] * vector.x + left.cells[13] * vector.y + left.cells[14] * vector.z + left.cells[15] * vector.w;

	return result;
}

__device__ Matrix quaternionToMatrix(float4 rotation)
{
	Matrix mat = createMatrix(1.0f);
	mat.cells[0] = 1 - 2 * rotation.y * rotation.y - 2 * rotation.z * rotation.z;
	mat.cells[1] = 2 * rotation.x*rotation.y - 2 * rotation.z*rotation.w;
	mat.cells[2] = 2 * rotation.x*rotation.z + 2 * rotation.y*rotation.w;
	mat.cells[3] = 0;

	mat.cells[4] = 2 * rotation.x*rotation.y + 2 * rotation.z*rotation.w;
	mat.cells[5] = 1 - 2 * rotation.x*rotation.x - 2 * rotation.z*rotation.z;
	mat.cells[6] = 2 * rotation.y*rotation.z - 2 * rotation.x*rotation.w;
	mat.cells[7] = 0;

	mat.cells[8] = 2 * rotation.x*rotation.z - 2 * rotation.y*rotation.w;
	mat.cells[9] = 2 * rotation.y*rotation.z + 2 * rotation.x * rotation.w;
	mat.cells[10] = 1 - 2 * rotation.x*rotation.x - 2 * rotation.y*rotation.y;
	mat.cells[11] = 0;

	mat.cells[12] = 0;
	mat.cells[13] = 0;
	mat.cells[14] = 0;
	mat.cells[15] = 1;

	return mat;

}

__device__ Matrix scaleMatrix(Matrix left, float3 scale)
{
	Matrix mat = createMatrix(1.0f);
	mat.cells[0] = scale.x;
	mat.cells[5] = scale.y;
	mat.cells[10] = scale.z;
	return multiplyMatrices(left, mat);
}

__device__ Matrix translateMatrix(Matrix left, float3 translation)
{
	Matrix mat = createMatrix(1.0f);
	mat.cells[3] = translation.x;
	mat.cells[7] = translation.y;
	mat.cells[11] = translation.z;

	return multiplyMatrices(left, mat);
}

__device__ Matrix rotateMatrixAlongX(Matrix left, float angle)
{
	Matrix mat = createMatrix(1.0f);
	mat.cells[5] = cosf(angle);
	mat.cells[6] = -sinf(angle);
	mat.cells[9] = sinf(angle);
	mat.cells[10] = cosf(angle);

	return multiplyMatrices(left, mat);
}

__device__ Matrix rotateMatrixAlongY(Matrix left, float angle)
{
	Matrix mat = createMatrix(1.0f);
	mat.cells[0] = cosf(angle);
	mat.cells[2] = sinf(angle);
	mat.cells[8] = -sinf(angle);
	mat.cells[10] = cosf(angle);

	return multiplyMatrices(left, mat);
}

__device__ Matrix rotateMatrixAlongZ(Matrix left, float angle)
{
	Matrix mat = createMatrix(1.0f);
	mat.cells[0] = cosf(angle);
	mat.cells[1] = -sinf(angle);
	mat.cells[4] = sinf(angle);
	mat.cells[5] = cosf(angle);

	return multiplyMatrices(left, mat);
}

__device__ Matrix rotateEuler(Matrix left, float x, float y, float z)
{
	left = rotateMatrixAlongX(left, x);
	left = rotateMatrixAlongY(left, y);
	left = rotateMatrixAlongZ(left, z);
	return left;
}

__device__ Matrix rotateMatrix(Matrix left, float4 quaternion)
{
	return multiplyMatrices(left, quaternionToMatrix(quaternion));
}

__device__ float magnitudeSqr(float3 vector)
{
	return (vector.x * vector.x) + (vector.y * vector.y) + (vector.z * vector.z);
}

__device__ float clamp(float value, float min, float max)
{
	return fminf(fmaxf(value, min), max);
}