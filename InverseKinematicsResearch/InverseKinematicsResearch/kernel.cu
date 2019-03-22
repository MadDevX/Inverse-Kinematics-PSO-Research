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
#include "utility_kernels.cuh"
#include "matrix_operations.cuh"
#include "ik_constants.h"

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

struct ParticleNew
{
	float positions[DEGREES_OF_FREEDOM];
	float velocities[DEGREES_OF_FREEDOM];
	float localBest[DEGREES_OF_FREEDOM];
};

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

//__global__ void simulateParticlesNewKernel()

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
