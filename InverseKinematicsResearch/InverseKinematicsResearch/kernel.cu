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


__constant__ float angleWeight = 3.0f;
__constant__ float errorThreshold = 0.1f;



__device__ Matrix calculateModelMatrix(NodeCUDA *chain, ParticleNew *particle, int nodeIndex)
{
	Matrix matrix = createMatrix(1.0f);
	while (nodeIndex != 0)
	{
		int particleIndex = (nodeIndex - 1) * 3;
		float3 particleEulerRotation = make_float3(particle->positions[particleIndex],
			particle->positions[particleIndex + 1],
			particle->positions[particleIndex + 2]);

		Matrix tempMat = createMatrix(1.0f);
		tempMat = rotateEuler(tempMat, particleEulerRotation);
		tempMat = translateMatrix(tempMat, make_float3(chain[nodeIndex].length, 0.0f, 0.0f));
		matrix = multiplyMatrices(tempMat,matrix);
		nodeIndex = chain[nodeIndex].parentIndex;
	}
	Matrix originMatrix = createMatrix(1.0f);
	originMatrix = translateMatrix(originMatrix, chain[nodeIndex].position);
	originMatrix = rotateEuler(originMatrix, chain[nodeIndex].rotation);
	return multiplyMatrices(originMatrix, matrix);
}


//Ewentualnie kolizje moga byc sprawdzane dla odcinka i colliderow, wtedy przekazujemy do funkcji 2 x float3 i liste colliderow.
//Wtedy checkCollision byloby wywolane w wewnetrznej petli calculateDistance.
__device__ bool checkCollisions(NodeCUDA *chain, ParticleNew particle /*, Collider* colliders*/)
{
	return false;
}

__device__ float calculateDistanceNew(NodeCUDA *chain, ParticleNew particle)
{
	float rotationDifference = 0.0f;
	float distance = 0.0f;

	for(int ind = 1; ind <= DEGREES_OF_FREEDOM / 3; ind++)
	{
		float3 chainRotation = chain[ind].rotation;
		float3 particleRotation = make_float3(
			particle.positions[(ind - 1) * 3],
			particle.positions[(ind - 1) * 3 + 1],
			particle.positions[(ind - 1) * 3 + 2]);

		rotationDifference = rotationDifference + magnitudeSqr(chainRotation - particleRotation);
		
		if (chain[ind].nodeType == NodeType::effectorNode)
		{		
			Matrix model = calculateModelMatrix(chain,&particle, ind);
			float4 position = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
			position = multiplyMatByVec(model, position);

			float distTmp = magnitudeSqr(make_float3(
													position.x - chain[ind].targetPosition.x,
													position.y - chain[ind].targetPosition.y,
													position.z - chain[ind].targetPosition.z));
		
			distance = distance + distTmp;
		}
		
		
	}

	return distance + angleWeight/(DEGREES_OF_FREEDOM / 3) * rotationDifference;
}

__global__ void simulateParticlesNewKernel(ParticleNew *particles, float *bests, curandState_t *randoms, int size, NodeCUDA *chain, Config config, CoordinatesNew global, float globalMin)
{
	extern __shared__ NodeCUDA sharedChain[];

	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = gridDim.x * blockDim.x;

	for (int i = id % blockDim.x; i < (DEGREES_OF_FREEDOM / 3) + 1; i++)
	{
		sharedChain[i] = chain[i];
	}

	for (int i = id; i < size; i += stride)
	{
		
		for (int deg = 0; deg < DEGREES_OF_FREEDOM; deg++)
		{
			particles[i].velocities[deg] = config._inertia * curand_uniform(&randoms[i]) * particles[i].velocities[deg] +
										   config._local   * curand_uniform(&randoms[i]) * (particles[i].localBest[deg] - particles[i].positions[deg]) +
										   config._global  * curand_uniform(&randoms[i]) * (global.positions[deg]- particles[i].positions[deg]);

			particles[i].positions[deg] += particles[i].velocities[deg];

			
		}

		//for (int ind = 1; ind <= DEGREES_OF_FREEDOM/3; ind++)
		//{
		//	int deg = (ind - 1) * 3;
		//	particles[i].positions[deg]   =   clamp(particles[i].positions[deg], chain[ind].minRotation.x, chain[ind].maxRotation.x);
		//	particles[i].positions[deg + 1] = clamp(particles[i].positions[deg+1], chain[ind].minRotation.y, chain[ind].maxRotation.y);
		//	particles[i].positions[deg + 2] = clamp(particles[i].positions[deg+2], chain[ind].minRotation.z, chain[ind].maxRotation.z);
		//}	
		float currentDistance = calculateDistanceNew(sharedChain, particles[i]);
		
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
	extern __shared__ NodeCUDA sharedChain[];

	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = gridDim.x * blockDim.x;

	for (int i = id % blockDim.x; i < (DEGREES_OF_FREEDOM / 3) + 1; i++)
	{
		sharedChain[i] = chain[i];
	}


	
	for (int i = id; i < size; i += stride)
	{
	
		for (int deg = 0; deg < DEGREES_OF_FREEDOM; deg += 3)
		{
			//Uniform distribution of particles across the domain
			int chainIndex = (deg / 3) + 1;
			float3 eulerMaxConstraint = sharedChain[chainIndex].maxRotation;
			float3 eulerMinConstraint = sharedChain[chainIndex].minRotation;

			//printf("maxconstraint x %f\n", chain[chainIndex].maxRotation.x);
			//printf("maxconstraint y %f\n", chain[chainIndex].maxRotation.y);
			//printf("maxconstraint z %f\n", chain[chainIndex].maxRotation.z);


			//printf("quaterniondiff - deg %d : %f\n",deg, eulerMaxConstraint.z - eulerMinConstraint.z);
			//printf("quaterniondiff - deg %d : %f\n",deg+1, eulerMaxConstraint.x - eulerMinConstraint.x);
			//printf("quaterniondiff - deg %d : %f\n",deg+2, eulerMaxConstraint.y - eulerMinConstraint.y);
			//particles[i].positions[deg] =     (curand_uniform(&randoms[i])    *6.28f - 3.14f); //(curand_uniform(&randoms[i]) * (eulerMaxConstraint.x - eulerMinConstraint.x)) + eulerMinConstraint.x;
			//particles[i].positions[deg + 1] = (curand_uniform(&randoms[i])*6.28f - 3.14f);// (curand_uniform(&randoms[i]) * (eulerMaxConstraint.y - eulerMinConstraint.y)) + eulerMinConstraint.y;
			//particles[i].positions[deg + 2] = (curand_uniform(&randoms[i])*6.28f - 3.14f);// (curand_uniform(&randoms[i]) * (eulerMaxConstraint.z - eulerMinConstraint.z)) + eulerMinConstraint.z;
			float3 eulerRot = sharedChain[chainIndex].rotation;
			particles[i].positions[deg] = eulerRot.x;
			particles[i].positions[deg + 1] = eulerRot.y;
			particles[i].positions[deg + 2] = eulerRot.z;

		}

		//Init bests with current data
		for (int deg = 0; deg < DEGREES_OF_FREEDOM; deg += 1)
		{
			particles[i].velocities[deg] = curand_uniform(&randoms[i]) * 2.0f - 1.0f;
			particles[i].localBest[deg] = particles[i].positions[deg];
		}

		//Calculate bests
		localBests[i] = calculateDistanceNew(sharedChain, particles[i]);
		
	}

}



cudaError_t calculatePSONew(ParticleNew *particles, float *bests, curandState_t *randoms, int size, NodeCUDA *chain, Config config, CoordinatesNew *result)
{
	cudaError_t status;
	CoordinatesNew global;
	float globalMin;
	int numBlocks = (size + blockSize - 1) / blockSize;
	int sharedMemorySize = sizeof(NodeCUDA)*((DEGREES_OF_FREEDOM / 3) + 1);

	initParticlesNewKernel << <numBlocks, blockSize, sharedMemorySize>> > (particles, bests, randoms, chain, size);
	checkCuda(status = cudaGetLastError());
	if (status != cudaSuccess) return status;
	checkCuda(status = cudaDeviceSynchronize());

	float *globalBest = thrust::min_element(thrust::host, bests, bests + size);

	int globalIndex = globalBest - bests;

	for (int deg = 0; deg < DEGREES_OF_FREEDOM; deg++)
	{
		global.positions[deg] = particles[globalIndex].localBest[deg];
	}
	
	globalMin = bests[globalIndex];

	for (int i = 0; i < config._iterations; i++)
	{
		simulateParticlesNewKernel << <numBlocks, blockSize, sharedMemorySize >> > (particles, bests, randoms, size, chain, config, global, globalMin);
		checkCuda(status = cudaGetLastError());
		if (status != cudaSuccess) return status;
		checkCuda(status = cudaDeviceSynchronize());
		globalBest = thrust::min_element(thrust::host, bests, bests + size);
		globalIndex = globalBest - bests;
		if (globalMin > bests[globalIndex])
		{
			globalMin = bests[globalIndex];
			for (int deg = 0; deg < DEGREES_OF_FREEDOM; deg++)
			{
				global.positions[deg] = particles[globalIndex].localBest[deg];
			}
		}
		for (int i = 0; i < size; i++)
		{
			//printf("\tODLGLOSC %d - %f: \n",i, bests[i]);
		}
		

	}

	*result = global;
	//printf("Global Min: %f; Index = %d\n", globalMin, globalIndex);
	return status;
}