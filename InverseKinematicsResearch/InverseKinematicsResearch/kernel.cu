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
#include "quaternion_operations.cuh"
#include "vector_operations.cuh"
#include "ik_constants.h"

__constant__ float locality = -0.1f;
__constant__ float angleWeight = 0.05f;
__constant__ float errorThreshold = 0.1f;

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

__device__ Matrix calculateModelMatrix(NodeCUDA *chain, int nodeIndex)
{
	if (nodeIndex == 0)
	{	
		Matrix matrix = createMatrix(1.0f);
		matrix = translateMatrix(matrix, chain[nodeIndex].position);
		matrix = rotateMatrix(matrix, chain[nodeIndex].rotation);
		return matrix;
	}
	else
	{
		Matrix matrix = calculateModelMatrix(chain, chain[nodeIndex].parentIndex);
		matrix = rotateMatrix(matrix, chain[nodeIndex].rotation);
		matrix = translateMatrix(matrix, make_float3(chain[nodeIndex].length,0.0f,0.0f));
		return matrix;
	}
}

__device__ float calculateDistanceNew(NodeCUDA *chain, ParticleNew particle)
{
	float quaternionDifference = 0.0f;
	float distance = 0.0f;

	for(int ind = 1; ind <= DEGREES_OF_FREEDOM / 3; ind++)
	{
		float4 chainQuaternion = chain[ind].rotation;
		float4 particleQuaternionRotation = eulerToQuaternion(make_float3(particle.positions[(ind - 1) * 3],
			particle.positions[(ind - 1) * 3 + 1],
			particle.positions[(ind - 1) * 3 + 2]));

		quaternionDifference = quaternionDifference + magnitudeSqr(chainQuaternion - particleQuaternionRotation);
		
		if (chain[ind].nodeType == NodeType::effectorNode)
		{		
			Matrix model = calculateModelMatrix(chain, ind);
			float4 position = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
			position = multiplyMatByVec(model, position);

			distance = distance + magnitudeSqr(make_float3(position.x - chain[ind].targetPosition.x,
												  position.y - chain[ind].targetPosition.y,
												  position.z - chain[ind].targetPosition.z));
		}
		
	}

	return distance + angleWeight * (quaternionDifference);
}

__global__ void simulateParticlesNewKernel(ParticleNew *particles, float *bests, curandState_t *randoms, int size, NodeCUDA *chain, Config config, CoordinatesNew global, float globalMin)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = gridDim.x * blockDim.x;

	for (int i = id; i < size; i += stride)
	{

		
		for (int deg = 0; deg < DEGREES_OF_FREEDOM; deg++)
		{
			particles[i].velocities[deg] = config._inertia * particles[i].velocities[deg] +
				config._local * curand_uniform(&randoms[i]) * (particles[i].localBest[deg] - particles[i].positions[deg]) +
				config._global * curand_uniform(&randoms[i]) * (global.positions[deg]- particles[i].positions[deg]);

			particles[i].positions[deg] += particles[i].velocities[deg];

			
		}

		for (int ind = 1; ind <= DEGREES_OF_FREEDOM/3; ind++)
		{
			int deg = (ind - 1) * 3;
			particles[i].positions[deg]   =   clamp(particles[i].positions[deg], chain[ind].minRotation.x, chain[ind].maxRotation.x);
			particles[i].positions[deg + 1] = clamp(particles[i].positions[deg+1], chain[ind].minRotation.y, chain[ind].maxRotation.y);
			particles[i].positions[deg + 2] = clamp(particles[i].positions[deg+2], chain[ind].minRotation.z, chain[ind].maxRotation.z);
		}	
		float currentDistance = calculateDistanceNew(chain, particles[i]);
		
		if (currentDistance < bests[i])
		{
			
			bests[i] = currentDistance;
			for (int deg = 0; deg < DEGREES_OF_FREEDOM; deg++)
			{
				particles[i].localBest[deg] = particles[i].positions[deg];
				
			}
			
		}
	}
}


__global__ void initParticlesNewKernel(ParticleNew *particles, float *localBests, curandState_t *randoms, NodeCUDA * chain, int size)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = gridDim.x * blockDim.x;

	for (int i = id; i < size; i += stride)
	{
		if (curand_uniform(&randoms[i]) > locality)
		{

			for (int deg = 0; deg < DEGREES_OF_FREEDOM; deg+=3)
			{

				float3 euler = quaternionToEuler(chain[(deg/3)+1].rotation);
				particles[i].positions[deg] = euler.x;
				particles[i].positions[deg+1] = euler.y;
				particles[i].positions[deg+2] = euler.z;		
			}
			
		}
		else
		{
			
			for (int deg = 0; deg < DEGREES_OF_FREEDOM; deg += 3)
			{
				//Uniform distribution of particles across the domain
				int chainIndex = (deg / 3) + 1;
				float3 eulerMaxConstraint = quaternionToEuler(chain[chainIndex].maxRotation);
				float3 eulerMinConstraint = quaternionToEuler(chain[chainIndex].minRotation);
				
				particles[i].positions[deg] = (curand_uniform(&randoms[i]) * (eulerMaxConstraint.x - eulerMinConstraint.x)) + eulerMinConstraint.x;
				particles[i].positions[deg+1] = (curand_uniform(&randoms[i]) * (eulerMaxConstraint.y - eulerMinConstraint.y)) + eulerMinConstraint.y;
				particles[i].positions[deg+2] = (curand_uniform(&randoms[i]) * (eulerMaxConstraint.z - eulerMinConstraint.z)) + eulerMinConstraint.z;
			}
		}

		//Init bests with current data
		for (int deg = 0; deg < DEGREES_OF_FREEDOM; deg += 1)
		{
			particles[i].velocities[deg] = curand_uniform(&randoms[i]) * 2.0f - 1.0f;
			particles[i].localBest[deg] = particles[i].positions[deg];
		}

		//Calculate bests
		localBests[i] = calculateDistanceNew(chain, particles[i]);
	}

}



cudaError_t calculatePSONew(ParticleNew *particles, float *bests, curandState_t *randoms, int size, NodeCUDA *chain, Config config, CoordinatesNew *result)
{
	cudaError_t status;
	CoordinatesNew global;
	float globalMin;
	int numBlocks = (size + blockSize - 1) / blockSize;

	initParticlesNewKernel << <numBlocks, blockSize >> > (particles, bests, randoms, chain, size);
	checkCuda(status = cudaGetLastError());
	if (status != cudaSuccess) return status;
	checkCuda(status = cudaDeviceSynchronize());


	float *globalBest = thrust::min_element(thrust::host, bests, bests + size);
	int globalIndex = globalBest - bests;
	printf("global index = %d \n", globalIndex);
	for (int deg = 0; deg < DEGREES_OF_FREEDOM; deg++)
	{
		global.positions[deg] = particles[globalIndex].localBest[deg];
	}
	
	globalMin = bests[globalIndex];

	for (int i = 0; i < config._iterations; i++)
	{
		simulateParticlesNewKernel << <numBlocks, blockSize >> > (particles, bests, randoms, size, chain, config, global, globalMin);
		checkCuda(status = cudaGetLastError());
		if (status != cudaSuccess) return status;
		checkCuda(status = cudaDeviceSynchronize());
		globalBest = thrust::min_element(thrust::host, bests, bests + size);
		globalIndex = globalBest - bests;
		for (int deg = 0; deg < DEGREES_OF_FREEDOM; deg++)
		{
			global.positions[deg] = particles[globalIndex].localBest[deg];
		}

		globalMin = bests[globalIndex];
	}

	*result = global;

	return status;
}