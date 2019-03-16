#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <curand_kernel.h>
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <thrust/extrema.h>
#include "Particle.h"

#define blockSize 256
__constant__ float locality = -0.1f;
__constant__ float angleWeight = 0.05f;
__constant__ float errorThreshold = 0.1f;
// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error: %s\n",
			cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}
#endif
	return result;
}

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

__device__ float magnitudeSqr(float3 vector)
{
	return (vector.x * vector.x) + (vector.y * vector.y) + (vector.z * vector.z);
}

__device__ float clamp(float value, float min, float max)
{
	return fminf(fmaxf(value, min), max);
}

__device__ float calculateDistance(KinematicChainCuda chain, Particle particle, float3 targetPosition)
{
	Matrix model = createMatrix(1.0f);
	model = translateMatrix(model, chain._shoulderPosition);
	model = rotateEuler(model, particle.positions.shoulderRotX, particle.positions.shoulderRotY, particle.positions.shoulderRotZ);
	model = translateMatrix(model, make_float3(chain._armLength, 0.0f, 0.0f));
	model = rotateEuler(model, particle.positions.elbowRotX, particle.positions.elbowRotY, particle.positions.elbowRotZ);
	model = translateMatrix(model, make_float3(chain._forearmLength, 0.0f, 0.0f));
	float4 position = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
	position = multiplyMatByVec(model, position);
	float3 diff = make_float3(position.x - targetPosition.x, position.y - targetPosition.y, position.z - targetPosition.z);
	float3 diffShoulder = make_float3(chain._shoulderRotation.x - particle.positions.shoulderRotX, chain._shoulderRotation.y - particle.positions.shoulderRotY, chain._shoulderRotation.z - particle.positions.shoulderRotZ);
	float3 diffElbow = make_float3(chain._elbowRotation.x - particle.positions.elbowRotX, chain._elbowRotation.y - particle.positions.elbowRotY, chain._elbowRotation.z - particle.positions.elbowRotZ);
	float distance = magnitudeSqr(diff);
	return distance + angleWeight * (magnitudeSqr(diffShoulder) + magnitudeSqr(diffElbow));
}

__global__ void simulateParticlesKernel(Particle *particles, float *bests, curandState_t *randoms, int size, KinematicChainCuda chain, float3 targetPosition, Config config, Coordinates global, float globalMin)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = gridDim.x * blockDim.x;

	for (int i = id; i < size; i += stride)
	{
		particles[i].velocities.shoulderRotX = config._inertia * particles[i].velocities.shoulderRotX +
											   config._local * curand_uniform(&randoms[i]) * (particles[i].localBest.shoulderRotX - particles[i].positions.shoulderRotX) +
											   config._global * curand_uniform(&randoms[i]) * (global.shoulderRotX - particles[i].positions.shoulderRotX);
		particles[i].velocities.shoulderRotY = config._inertia * particles[i].velocities.shoulderRotY +
											   config._local * curand_uniform(&randoms[i]) * (particles[i].localBest.shoulderRotY - particles[i].positions.shoulderRotY) +
											   config._global * curand_uniform(&randoms[i]) * (global.shoulderRotY - particles[i].positions.shoulderRotY);
		particles[i].velocities.shoulderRotZ = config._inertia * particles[i].velocities.shoulderRotZ +
											   config._local * curand_uniform(&randoms[i]) * (particles[i].localBest.shoulderRotZ - particles[i].positions.shoulderRotZ) +
											   config._global * curand_uniform(&randoms[i]) * (global.shoulderRotZ - particles[i].positions.shoulderRotZ);
		particles[i].velocities.elbowRotX =    config._inertia * particles[i].velocities.elbowRotX +
											   config._local * curand_uniform(&randoms[i]) * (particles[i].localBest.elbowRotX - particles[i].positions.elbowRotX) +
											   config._global * curand_uniform(&randoms[i]) * (global.elbowRotX - particles[i].positions.elbowRotX);
		particles[i].velocities.elbowRotY =    config._inertia * particles[i].velocities.elbowRotY +
											   config._local * curand_uniform(&randoms[i]) * (particles[i].localBest.elbowRotY - particles[i].positions.elbowRotY) +
											   config._global * curand_uniform(&randoms[i]) * (global.elbowRotY - particles[i].positions.elbowRotY);
		particles[i].velocities.elbowRotZ =    config._inertia * particles[i].velocities.elbowRotZ +
											   config._local * curand_uniform(&randoms[i]) * (particles[i].localBest.elbowRotZ - particles[i].positions.elbowRotZ) +
											   config._global * curand_uniform(&randoms[i]) * (global.elbowRotZ - particles[i].positions.elbowRotZ);

		particles[i].positions.shoulderRotX += particles[i].velocities.shoulderRotX;
		particles[i].positions.shoulderRotY += particles[i].velocities.shoulderRotY;
		particles[i].positions.shoulderRotZ += particles[i].velocities.shoulderRotZ;
		particles[i].positions.elbowRotX += particles[i].velocities.elbowRotX;
		particles[i].positions.elbowRotY += particles[i].velocities.elbowRotY;
		particles[i].positions.elbowRotZ += particles[i].velocities.elbowRotZ;

		particles[i].positions.shoulderRotX = clamp(particles[i].positions.shoulderRotX, chain._minShoulder.x, chain._maxShoulder.x);
		particles[i].positions.shoulderRotY	= clamp(particles[i].positions.shoulderRotY, chain._minShoulder.y, chain._maxShoulder.y);
		particles[i].positions.shoulderRotZ	= clamp(particles[i].positions.shoulderRotZ, chain._minShoulder.z, chain._maxShoulder.z);
		particles[i].positions.elbowRotX	= clamp(particles[i].positions.elbowRotX, chain._minElbow.x, chain._maxElbow.x);
		particles[i].positions.elbowRotY	= clamp(particles[i].positions.elbowRotY, chain._minElbow.y, chain._maxElbow.y);
		particles[i].positions.elbowRotZ	= clamp(particles[i].positions.elbowRotZ, chain._minElbow.z, chain._maxElbow.z);


		float currentDistance = calculateDistance(chain, particles[i], targetPosition);
		if (currentDistance < bests[i])
		{
			bests[i] = currentDistance;
			particles[i].localBest = particles[i].positions;
		}
	}
}

__global__ void initParticlesKernel(Particle *particles, float *localBests, curandState_t *randoms, KinematicChainCuda chain, float3 targetPosition, int size)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = gridDim.x * blockDim.x;

	for (int i = id; i < size; i += stride)
	{
		if (curand_uniform(&randoms[i]) > locality)
		{
			particles[i].positions.shoulderRotX = chain._shoulderRotation.x;
			particles[i].positions.shoulderRotY = chain._shoulderRotation.y;
			particles[i].positions.shoulderRotZ = chain._shoulderRotation.z;
			particles[i].positions.elbowRotX = chain._elbowRotation.x;
			particles[i].positions.elbowRotY = chain._elbowRotation.y;
			particles[i].positions.elbowRotZ = chain._elbowRotation.z;
		}
		else
		{
			particles[i].positions.shoulderRotX = (curand_uniform(&randoms[i]) * (chain._maxShoulder.x - chain._minShoulder.x) + chain._minShoulder.x);
			particles[i].positions.shoulderRotY = (curand_uniform(&randoms[i]) * (chain._maxShoulder.y - chain._minShoulder.y) + chain._minShoulder.y);
			particles[i].positions.shoulderRotZ = (curand_uniform(&randoms[i]) * (chain._maxShoulder.z - chain._minShoulder.z) + chain._minShoulder.z);
			particles[i].positions.elbowRotX = (curand_uniform(&randoms[i]) * (chain._maxElbow.x - chain._minElbow.x) + chain._minElbow.x);
			particles[i].positions.elbowRotY = (curand_uniform(&randoms[i]) * (chain._maxElbow.y - chain._minElbow.y) + chain._minElbow.y);
			particles[i].positions.elbowRotZ = (curand_uniform(&randoms[i]) * (chain._maxElbow.z - chain._minElbow.z) + chain._minElbow.z);
		}

		particles[i].velocities.shoulderRotX = curand_uniform(&randoms[i]) * 2.0f - 1.0f;
		particles[i].velocities.shoulderRotY = curand_uniform(&randoms[i]) * 2.0f - 1.0f;
		particles[i].velocities.shoulderRotZ = curand_uniform(&randoms[i]) * 2.0f - 1.0f;
		particles[i].velocities.elbowRotX = curand_uniform(&randoms[i]) * 2.0f - 1.0f;
		particles[i].velocities.elbowRotY = curand_uniform(&randoms[i]) * 2.0f - 1.0f;
		particles[i].velocities.elbowRotZ = curand_uniform(&randoms[i]) * 2.0f - 1.0f;


		particles[i].localBest.shoulderRotX = particles[i].positions.shoulderRotX;
		particles[i].localBest.shoulderRotY = particles[i].positions.shoulderRotY;
		particles[i].localBest.shoulderRotZ = particles[i].positions.shoulderRotZ;
		particles[i].localBest.elbowRotX = particles[i].positions.elbowRotX;
		particles[i].localBest.elbowRotY = particles[i].positions.elbowRotY;
		particles[i].localBest.elbowRotZ = particles[i].positions.elbowRotZ;

		localBests[i] = calculateDistance(chain, particles[i], targetPosition);
	}
}

__global__ void randInitKernel(curandState_t *randoms, int size)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = gridDim.x * blockDim.x;

	for (int i = id; i < size; i += stride)
	{
		curand_init(i, 0, 0, &randoms[i]);
	}
}

cudaError_t initGenerators(curandState_t *randoms, int size)
{
	cudaError_t cudaStatus;

	int numBlocks = (size + blockSize - 1) / blockSize;

	randInitKernel << <numBlocks, blockSize >> > (randoms, size);

	checkCuda(cudaStatus = cudaGetLastError());

	if (cudaStatus == cudaSuccess)
		checkCuda(cudaStatus = cudaDeviceSynchronize());

	return cudaStatus;
}

cudaError_t calculatePSO(Particle *particles, float *bests, curandState_t *randoms, int size, KinematicChainCuda chain, float3 targetPosition, Config config, Coordinates *result)
{
	cudaError_t status;
	int numBlocks = (size + blockSize - 1) / blockSize;
	initParticlesKernel<<<numBlocks, blockSize>>>(particles, bests, randoms, chain, targetPosition, size);
	checkCuda(status = cudaGetLastError());
	if (status != cudaSuccess) return status;
	checkCuda(status = cudaDeviceSynchronize());

	Coordinates global;
	float globalMin;
	float *globalBest = thrust::min_element(thrust::host, bests, bests + size);
	int globalIndex = globalBest - bests;
	global = particles[globalIndex].localBest;
	globalMin = bests[globalIndex];
	for (int i = 0; i < config._iterations; i++)
	{
		simulateParticlesKernel<<<numBlocks, blockSize>>>(particles, bests, randoms, size, chain, targetPosition, config, global, globalMin);
		checkCuda(status = cudaGetLastError());
		if (status != cudaSuccess) return status;
		checkCuda(status = cudaDeviceSynchronize());

		globalBest = thrust::min_element(thrust::host, bests, bests + size);
		globalIndex = globalBest - bests;
		global = particles[globalIndex].localBest;
		globalMin = bests[globalIndex];
	}

	*result = global;

	return status;
}
