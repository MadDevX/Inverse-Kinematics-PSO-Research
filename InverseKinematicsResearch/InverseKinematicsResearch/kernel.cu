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
#include "GJKIntersection.cuh"
#include "ik_constants.h"

__constant__ float angleWeight = 3.0f;
__constant__ float errorThreshold = 0.1f;

__host__ __device__ int getParticleIndex(int particleCount, int particleIdx, ParticleProperty propType, int dimension)
{
	int idx = particleIdx + particleCount * dimension;
	if (propType == velocity)
	{
		idx += particleCount * DEGREES_OF_FREEDOM;
	}
	else if (propType == localBest)
	{
		idx += 2 * particleCount * DEGREES_OF_FREEDOM;
	}
	return idx;
}

__device__ void updateChainMatrices(NodeCUDA *chain, int particleCount, float* particles, int particleIdx, Matrix *matrices)
{
	int nodeCount = NODE_COUNT;
	int nodeIndex = 0;

	Matrix matrix = createMatrix(1.0f);
	matrix = translateMatrix(matrix, chain[nodeIndex].position);
	matrix = rotateEuler(matrix, chain[nodeIndex].rotation);
	for (int i = 0; i < 16; i++)
	{
		matrices[nodeIndex].cells[i] = matrix.cells[i];
	}

	for(nodeIndex = 1; nodeIndex < nodeCount; nodeIndex++)
	{
		int dimensionIdx = (nodeIndex - 1) * 3;
		int positionIdx = getParticleIndex(particleCount, particleIdx, position, dimensionIdx);
		float3 particleEulerRotation = make_float3(particles[positionIdx],
												   particles[positionIdx + particleCount],
												   particles[positionIdx + particleCount * 2]);
									///make_float3(particle.positions[dimensionIndex],
									///			   particle.positions[dimensionIndex + 1],
									///			   particle.positions[dimensionIndex + 2]);

		Matrix tempMat = createMatrix(1.0f);
		tempMat = rotateEuler(tempMat, particleEulerRotation);
		tempMat = translateMatrix(tempMat, make_float3(chain[nodeIndex].length, 0.0f, 0.0f));
		int parentIdx = chain[nodeIndex].parentIndex;
		matrix = multiplyMatrices(matrices[parentIdx], tempMat);
		for (int i = 0; i < 16; i++)
		{
			matrices[nodeIndex].cells[i] = matrix.cells[i];
		}
	}
}

__device__ float calculateDistance(NodeCUDA *chain, int particleCount, float* particles, int particleIdx, obj_t* colliders, int colliderCount)
{
	float rotationDifference = 0.0f;
	float distance = 0.0f;
	int nodeCount = NODE_COUNT;
	Matrix matrices[NODE_COUNT];
	updateChainMatrices(chain, particleCount, particles, particleIdx, matrices);
	for(int ind = 1; ind < nodeCount; ind++)
	{
		float3 chainRotation = chain[ind].rotation;
		int dimensionIdx = (ind - 1) * 3;
		int positionIdx = getParticleIndex(particleCount, particleIdx, position, dimensionIdx);
		float3 particleRotation = make_float3(particles[positionIdx],
											  particles[positionIdx + particleCount],
											  particles[positionIdx + particleCount * 2]);
								///make_float3(
								///particle.positions[dimensionIdx],
								///particle.positions[dimensionIdx + 1],
								///particle.positions[dimensionIdx + 2]);

		rotationDifference = rotationDifference + magnitudeSqr(chainRotation - particleRotation);

		float4 originVector = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
		Matrix model;
		for (int i = 0; i < 16; i++)
		{
			model.cells[i] = matrices[ind].cells[i];
		}
		float4 position = multiplyMatByVec(model, originVector);
		float4 rotation = matrixToQuaternion(model);

		obj_t nodeCollider;
		nodeCollider.pos = make_float3(position.x, position.y, position.z);
		nodeCollider.quat = rotation;
		nodeCollider.x = nodeCollider.y = nodeCollider.z = GIZMO_SIZE;


		obj_t linkCollider;
		float4 startPos = multiplyMatByVec(model, originVector); //this node
		float4 endPos = multiplyMatByVec(matrices[chain[ind].parentIndex], originVector); //parent node
		float4 centerPos = (startPos + endPos) * 0.5f;
		linkCollider.pos = make_float3(centerPos.x, centerPos.y, centerPos.z);
		linkCollider.quat = rotation;
		linkCollider.x = chain[ind].length;
		linkCollider.y = linkCollider.z = GIZMO_SIZE * 0.25f;

		GJKData_t gjkData;
		CCD_INIT(&gjkData);
		gjkData.max_iterations = GJK_ITERATIONS;
		int intersects = 0;
		for (int i = 0; i < colliderCount; i++)
		{
			intersects = GJKIntersect(&nodeCollider, &colliders[i], &gjkData);
			if (intersects)
			{
				return FLT_MAX;
			}
			intersects = GJKIntersect(&linkCollider, &colliders[i], &gjkData);
			if (intersects)
			{
				return FLT_MAX;
			}
		}


		if (chain[ind].nodeType == NodeType::effectorNode)
		{
			float distTmp = magnitudeSqr(make_float3(
													position.x - chain[ind].targetPosition.x,
													position.y - chain[ind].targetPosition.y,
													position.z - chain[ind].targetPosition.z));
		
			distance = distance + distTmp;
		}
		
		
	}

	return distance + angleWeight/(DEGREES_OF_FREEDOM / 3) * rotationDifference;
}

__global__ void simulateParticlesKernel(float *particles, float *localBests, curandState_t *randoms, int size, NodeCUDA *chain, Config config, Coordinates *global, float globalMin, obj_t* colliders, int colliderCount)
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
			int velocityIdx = getParticleIndex(size, i, velocity, deg);
			int positionIdx = getParticleIndex(size, i, position, deg);
			particles[velocityIdx] = config._inertia * curand_uniform(&randoms[i]) * particles[velocityIdx] +
									 config._local   * curand_uniform(&randoms[i]) * (particles[getParticleIndex(size, i, localBest, deg)] - particles[positionIdx]) +
									 config._global  * curand_uniform(&randoms[i]) * (global->positions[deg] - particles[positionIdx]);
			
			particles[positionIdx] += particles[velocityIdx];
			///particles[i].velocities[deg] = config._inertia * curand_uniform(&randoms[i]) * particles[i].velocities[deg] +
			///							   config._local   * curand_uniform(&randoms[i]) * (particles[i].localBest[deg] - particles[i].positions[deg]) +
			///							   config._global  * curand_uniform(&randoms[i]) * (global.positions[deg]- particles[i].positions[deg]);
			///
			///particles[i].positions[deg] += particles[i].velocities[deg];

			
		}

		//for (int ind = 1; ind <= DEGREES_OF_FREEDOM/3; ind++)
		//{
		//	int deg = (ind - 1) * 3;
		//	particles[i].positions[deg]   =   clamp(particles[i].positions[deg], chain[ind].minRotation.x, chain[ind].maxRotation.x);
		//	particles[i].positions[deg + 1] = clamp(particles[i].positions[deg+1], chain[ind].minRotation.y, chain[ind].maxRotation.y);
		//	particles[i].positions[deg + 2] = clamp(particles[i].positions[deg+2], chain[ind].minRotation.z, chain[ind].maxRotation.z);
		//}	
		float currentDistance = calculateDistance(sharedChain, size, particles, i, colliders, colliderCount);
		
		if (currentDistance < localBests[i])
		{
			
			localBests[i] = currentDistance;
			for (int deg = 0; deg < DEGREES_OF_FREEDOM; deg++)
			{
				particles[getParticleIndex(size, i, localBest, deg)] = particles[getParticleIndex(size, i, position, deg)];
				///particles[i].localBest[deg] = particles[i].positions[deg];
			}
			
		}
	}
}

__global__ void initParticlesKernel(float *particles, float *localBests, curandState_t *randoms, NodeCUDA * chain, int particleCount, obj_t* colliders, int colliderCount)
{
	extern __shared__ NodeCUDA sharedChain[];

	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = gridDim.x * blockDim.x;

	for (int i = id % blockDim.x; i < (DEGREES_OF_FREEDOM / 3) + 1; i++)
	{
		sharedChain[i] = chain[i];
	}


	
	for (int i = id; i < particleCount; i += stride)
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
			int positionIdx = getParticleIndex(particleCount, i, position, deg);
			particles[positionIdx]						= eulerRot.x;
			particles[positionIdx + particleCount]		= eulerRot.y;
			particles[positionIdx + particleCount * 2]	= eulerRot.z;

			///particles[i].positions[deg] = eulerRot.x;
			///particles[i].positions[deg + 1] = eulerRot.y;
			///particles[i].positions[deg + 2] = eulerRot.z;

		}

		//Init bests with current data
		for (int deg = 0; deg < DEGREES_OF_FREEDOM; deg += 1)
		{
			int positionIdx = getParticleIndex(particleCount, i, position, deg);
			particles[positionIdx + particleCount * DEGREES_OF_FREEDOM] = curand_uniform(&randoms[i]) * 2.0f - 1.0f;
			particles[positionIdx + particleCount * DEGREES_OF_FREEDOM * 2] = particles[positionIdx];
			///particles[i].velocities[deg] = curand_uniform(&randoms[i]) * 2.0f - 1.0f;
			///particles[i].localBest[deg] = particles[i].positions[deg];
		}

		//Calculate bests
		localBests[i] = calculateDistance(sharedChain, particleCount, particles, i, colliders, colliderCount);
		
	}

}

__global__ void updateGlobalBestCoordsKernel(float *particles, int particleCount, Coordinates* global, int globalIndex)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = gridDim.x * blockDim.x;

	for (int deg = id; deg < DEGREES_OF_FREEDOM; deg += stride)
	{
		global->positions[deg] = particles[getParticleIndex(particleCount, globalIndex, localBest, deg)];
		///global.positions[deg] = particles[globalIndex].localBest[deg];
	}
}

cudaError_t calculatePSO(float* particles, float* bests, curandState_t *randoms, int size, NodeCUDA *chain, Config config, Coordinates *result, obj_t* colliders, int colliderCount)
{
	cudaError_t status;
	float globalMin;
	float currentGlobalMin;
	int numBlocks = (size + blockSize - 1) / blockSize;
	int globalUpdateNumBlocks = (DEGREES_OF_FREEDOM + blockSize - 1) / blockSize;
	int sharedMemorySize = sizeof(NodeCUDA)*((DEGREES_OF_FREEDOM / 3) + 1);

	initParticlesKernel<<<numBlocks, blockSize, sharedMemorySize>>>(particles, bests, randoms, chain, size, colliders, colliderCount);
	checkCuda(status = cudaGetLastError());
	if (status != cudaSuccess) return status;
	//checkCuda(status = cudaDeviceSynchronize());//TODO

	float *globalBest = thrust::min_element(thrust::device, bests, bests + size);

	int globalIndex = globalBest - bests;

	updateGlobalBestCoordsKernel<<<globalUpdateNumBlocks, blockSize>>>(particles, size, result, globalIndex);
	checkCuda(status = cudaDeviceSynchronize());//TODO

	checkCuda(status = cudaMemcpy(&globalMin, bests + globalIndex, sizeof(float), cudaMemcpyDeviceToHost));
	///globalMin = bests[globalIndex];

	for (int i = 0; i < config._iterations; i++)
	{
		simulateParticlesKernel<<<numBlocks, blockSize, sharedMemorySize>>>(particles, bests, randoms, size, chain, config, result, globalMin, colliders, colliderCount);
		checkCuda(status = cudaGetLastError());
		if (status != cudaSuccess) return status;
		//checkCuda(status = cudaDeviceSynchronize()); //TODO
		globalBest = thrust::min_element(thrust::device, bests, bests + size);
		globalIndex = globalBest - bests;
		checkCuda(status = cudaMemcpy(&currentGlobalMin, bests + globalIndex, sizeof(float), cudaMemcpyDeviceToHost));
		if (globalMin > currentGlobalMin)
		{
			checkCuda(status = cudaMemcpy(&globalMin, bests + globalIndex, sizeof(float), cudaMemcpyDeviceToHost));
			updateGlobalBestCoordsKernel<<<globalUpdateNumBlocks, blockSize>>>(particles, size, result, globalIndex);
			checkCuda(status = cudaDeviceSynchronize()); //TODO
		}
	}

	return status;
}

#pragma region implementheaders

#pragma region vectorOperations
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
	dot = v1->x * v2->x;
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
#pragma endregion


__device__  void SupportCopy(support_t *d, const support_t *s)
{
	*d = *s;
}

__device__ void SimplexInit(simplex_t *s)
{
	s->last = -1;
}

__device__ int SimplexSize(const simplex_t *s)
{
	return s->last + 1;
}

__device__ const support_t *SimplexLast(const simplex_t *s)
{
	return SimplexPoint(s, s->last);
}

__device__ const support_t *SimplexPoint(const simplex_t *s, int idx)
{
	// here is no check on boundaries
	return &s->ps[idx];
}
__device__ support_t *SimplexPointW(simplex_t *s, int idx)
{
	return &s->ps[idx];
}

__device__ void SimplexAdd(simplex_t *s, const support_t *v)
{
	// here is no check on boundaries in sake of speed
	++s->last;
	SupportCopy(s->ps + s->last, v);
}

__device__ void SimplexSet(simplex_t *s, size_t pos, const support_t *a)
{
	SupportCopy(s->ps + pos, a);
}

__device__ void SimplexSetSize(simplex_t *s, int size)
{
	s->last = size - 1;
}

__device__ void SimplexSwap(simplex_t *s, size_t pos1, size_t pos2)
{
	support_t supp;

	SupportCopy(&supp, &s->ps[pos1]);
	SupportCopy(&s->ps[pos1], &s->ps[pos2]);
	SupportCopy(&s->ps[pos2], &supp);
}

__device__ void firstDir(const void *obj1, const void *obj2, float3 *dir) {
	dir->x = ONE;
	dir->y = ONE;
	dir->z = ZERO;
}

__device__ void supportBox(const void *_obj, const float3 *_dir, float3 *v)
{

	// assume that obj_t is user-defined structure that holds info about
	// object (in this case box: x, y, z, pos, quat - dimensions of box,
	// position and rotation)
	obj_t *obj = (obj_t *)_obj;
	float3 dir;
	float4 qinv;

	// apply rotation on direction vector
	float3Copy(&dir, _dir);
	quatInvert2(&qinv, &obj->quat);
	quatRotVec(&dir, &qinv);

	// compute support point in specified direction
	*v = make_float3(
		Signum(dir.x) * obj->x * 0.5f,
		Signum(dir.y) * obj->y * 0.5f,
		Signum(dir.z) * obj->z * 0.5f);
	// czlowiek to kubek q.e.d.
	// transform support point according to position and rotation of object
	quatRotVec(v, &obj->quat);
	float3Add(v, v, &obj->pos);

}

__device__ int GJKIntersect(const void *obj1, const void *obj2, const GJKData_t *data)
{
	simplex_t simplex;
	return GJK(obj1, obj2, data, &simplex) == 0;
}

__device__ static int GJK(const void *obj1, const void *obj2,
	const GJKData_t *data, simplex_t *simplex)
{
	unsigned long iterations;
	float3 dir; // direction vector
	support_t last; // last support point
	int do_simplex_res;

	// initialize simplex struct
	SimplexInit(simplex);

	// get first direction
	firstDir(obj1, obj2, &dir);
	// get first support point
	SupportCalc(obj1, obj2, &dir, &last);
	// and add this point to simplex as last one
	SimplexAdd(simplex, &last);

	// set up direction vector to as (O - last) which is exactly -last
	float3Copy(&dir, &last.v);
	float3Scale(&dir, -ONE);

	// start iterations
	for (iterations = 0UL; iterations < data->max_iterations; ++iterations) {
		// obtain support point
		SupportCalc(obj1, obj2, &dir, &last);

		// check if farthest point in Minkowski difference in direction dir
		// isn't somewhere before origin (the test on negative dot product)
		// - because if it is, objects are not intersecting at all.
		if (float3Dot(&last.v, &dir) < ZERO) {
			return -1; // intersection not found
		}

		// add last support vector to simplex
		SimplexAdd(simplex, &last);

		// if doSimplex returns 1 if objects intersect, -1 if objects don't
		// intersect and 0 if algorithm should continue
		do_simplex_res = doSimplex(simplex, &dir);
		if (do_simplex_res == 1) {
			return 0; // intersection found
		}
		else if (do_simplex_res == -1) {
			return -1; // intersection not found
		}

		if (IsZERO(float3Len(&dir))) {
			return -1; // intersection not found
		}
	}

	// intersection wasn't found
	return -1;
}

__device__ void SupportCalc(const void *obj1, const void *obj2, const float3 *_dir, support_t *supp)
{
	float3 dir;

	float3Copy(&dir, _dir);

	//wklejenie w v1 wyniku wywolania funkcji support1
	supportBox(obj1, &dir, &supp->v1);

	float3Scale(&dir, -ONE);
	//wklejenie w v2 wyniku wywolania funkcji support2
	supportBox(obj2, &dir, &supp->v2);

	//roznica minkowskiego
	float3Sub(&supp->v, &supp->v1, &supp->v2);
}


#pragma region doSimplexi

__device__ static int doSimplex(simplex_t *simplex, float3 *dir)
{
	if (SimplexSize(simplex) == 2) {
		// simplex contains segment only one segment
		return doSimplex2(simplex, dir);
	}
	else if (SimplexSize(simplex) == 3) {
		// simplex contains triangle
		return doSimplex3(simplex, dir);
	}
	else { // ccdSimplexSize(simplex) == 4
	   // tetrahedron - this is the only shape which can encapsule origin
	   // so doSimplex4() also contains test on it
		return doSimplex4(simplex, dir);
	}
}

__device__ static int doSimplex2(simplex_t *simplex, float3 *dir)
{
	const support_t *A, *B;
	float3 AB, AO, tmp;
	float dot;

	// get last added as A
	A = SimplexLast(simplex);
	// get the other point
	B = SimplexPoint(simplex, 0);
	// compute AB oriented segment
	float3Sub(&AB, &B->v, &A->v);
	// compute AO vector
	float3Copy(&AO, &A->v);
	float3Scale(&AO, -ONE);

	// dot product AB . AO
	dot = float3Dot(&AB, &AO);

	// check if origin doesn't lie on AB segment
	float3Cross(&tmp, &AB, &AO);
	if (IsZERO(float3Len(&tmp)) && dot > ZERO) {
		return 1;
	}

	// check if origin is in area where AB segment is
	if (IsZERO(dot) || dot < ZERO) {
		// origin is in outside are of A
		SimplexSet(simplex, 0, A);
		SimplexSetSize(simplex, 1);
		float3Copy(dir, &AO);
	}
	else {
		// origin is in area where AB segment is

		// keep simplex untouched and set direction to
		// AB x AO x AB
		tripleCross(&AB, &AO, &AB, dir);
	}

	return 0;
}

__device__ static int doSimplex3(simplex_t *simplex, float3 *dir)
{
	const float3 origin = make_float3(0.f, 0.f, 0.f);
	const float3* originPtr = &origin;

	const support_t *A, *B, *C;
	float3 AO, AB, AC, ABC, tmp;
	float dot, dist;

	// get last added as A
	A = SimplexLast(simplex);
	// get the other points
	B = SimplexPoint(simplex, 1);
	C = SimplexPoint(simplex, 0);

	// check touching contact
	dist = Vec3PointTriDist2(originPtr, &A->v, &B->v, &C->v, NULL);
	if (IsZERO(dist)) {
		return 1;
	}

	// check if triangle is really triangle (has area > 0)
	// if not simplex can't be expanded and thus no itersection is found
	if (float3Eq(&A->v, &B->v) || float3Eq(&A->v, &C->v)) {
		return -1;
	}

	// compute AO vector
	float3Copy(&AO, &A->v);
	float3Scale(&AO, -ONE);

	// compute AB and AC segments and ABC vector (perpendircular to triangle)
	float3Sub(&AB, &B->v, &A->v);
	float3Sub(&AC, &C->v, &A->v);
	float3Cross(&ABC, &AB, &AC);

	float3Cross(&tmp, &ABC, &AC);
	dot = float3Dot(&tmp, &AO);
	if (IsZERO(dot) || dot > ZERO) {
		dot = float3Dot(&AC, &AO);
		if (IsZERO(dot) || dot > ZERO) {
			// C is already in place
			SimplexSet(simplex, 1, A);
			SimplexSetSize(simplex, 2);
			tripleCross(&AC, &AO, &AC, dir);
		}
		else {
			dot = float3Dot(&AB, &AO);
			if (IsZERO(dot) || dot > ZERO) {
				SimplexSet(simplex, 0, B);
				SimplexSet(simplex, 1, A);
				SimplexSetSize(simplex, 2);
				tripleCross(&AB, &AO, &AB, dir);
			}
			else {
				SimplexSet(simplex, 0, A);
				SimplexSetSize(simplex, 1);
				float3Copy(dir, &AO);
			}
		}
	}
	else {
		float3Cross(&tmp, &AB, &ABC);
		dot = float3Dot(&tmp, &AO);
		if (IsZERO(dot) || dot > ZERO) {
			dot = float3Dot(&AB, &AO);
			if (IsZERO(dot) || dot > ZERO) {
				SimplexSet(simplex, 0, B);
				SimplexSet(simplex, 1, A);
				SimplexSetSize(simplex, 2);
				tripleCross(&AB, &AO, &AB, dir);
			}
			else {
				SimplexSet(simplex, 0, A);
				SimplexSetSize(simplex, 1);
				float3Copy(dir, &AO);
			}
		}
		else {
			dot = float3Dot(&ABC, &AO);
			if (IsZERO(dot) || dot > ZERO) {
				float3Copy(dir, &ABC);
			}
			else {
				support_t Ctmp;
				SupportCopy(&Ctmp, C);
				SimplexSet(simplex, 0, B);
				SimplexSet(simplex, 1, &Ctmp);

				float3Copy(dir, &ABC);
				float3Scale(dir, -ONE);
			}
		}
	}

	return 0;
}

__device__ static int doSimplex4(simplex_t *simplex, float3 *dir)
{
	const float3 origin = make_float3(0.f, 0.f, 0.f);
	const float3* originPtr = &origin;

	const support_t *A, *B, *C, *D;
	float3 AO, AB, AC, AD, ABC, ACD, ADB;
	int B_on_ACD, C_on_ADB, D_on_ABC;
	int AB_O, AC_O, AD_O;
	float dist;

	// get last added as A
	A = SimplexLast(simplex);
	// get the other points
	B = SimplexPoint(simplex, 2);
	C = SimplexPoint(simplex, 1);
	D = SimplexPoint(simplex, 0);

	// check if tetrahedron is really tetrahedron (has volume > 0)
	// if it is not simplex can't be expanded and thus no intersection is
	// found
	dist = Vec3PointTriDist2(&A->v, &B->v, &C->v, &D->v, NULL);
	if (IsZERO(dist)) {
		return -1;
	}

	// check if origin lies on some of tetrahedron's face - if so objects
	// intersect
	dist = Vec3PointTriDist2(originPtr, &A->v, &B->v, &C->v, NULL);
	if (IsZERO(dist))
		return 1;
	dist = Vec3PointTriDist2(originPtr, &A->v, &C->v, &D->v, NULL);
	if (IsZERO(dist))
		return 1;
	dist = Vec3PointTriDist2(originPtr, &A->v, &B->v, &D->v, NULL);
	if (IsZERO(dist))
		return 1;
	dist = Vec3PointTriDist2(originPtr, &B->v, &C->v, &D->v, NULL);
	if (IsZERO(dist))
		return 1;

	// compute AO, AB, AC, AD segments and ABC, ACD, ADB normal vectors
	float3Copy(&AO, &A->v);
	float3Scale(&AO, -ONE);
	float3Sub(&AB, &B->v, &A->v);
	float3Sub(&AC, &C->v, &A->v);
	float3Sub(&AD, &D->v, &A->v);
	float3Cross(&ABC, &AB, &AC);
	float3Cross(&ACD, &AC, &AD);
	float3Cross(&ADB, &AD, &AB);

	// side (positive or negative) of B, C, D relative to planes ACD, ADB
	// and ABC respectively
	B_on_ACD = Signum(float3Dot(&ACD, &AB));
	C_on_ADB = Signum(float3Dot(&ADB, &AC));
	D_on_ABC = Signum(float3Dot(&ABC, &AD));

	// whether origin is on same side of ACD, ADB, ABC as B, C, D
	// respectively
	AB_O = Signum(float3Dot(&ACD, &AO)) == B_on_ACD;
	AC_O = Signum(float3Dot(&ADB, &AO)) == C_on_ADB;
	AD_O = Signum(float3Dot(&ABC, &AO)) == D_on_ABC;

	if (AB_O && AC_O && AD_O) {
		// origin is in tetrahedron
		return 1;

		// rearrange simplex to triangle and call doSimplex3()
	}
	else if (!AB_O) {
		// B is farthest from the origin among all of the tetrahedron's
		// points, so remove it from the list and go on with the triangle
		// case

		// D and C are in place
		SimplexSet(simplex, 2, A);
		SimplexSetSize(simplex, 3);
	}
	else if (!AC_O) {
		// C is farthest
		SimplexSet(simplex, 1, D);
		SimplexSet(simplex, 0, B);
		SimplexSet(simplex, 2, A);
		SimplexSetSize(simplex, 3);
	}
	else { // (!AD_O)
		SimplexSet(simplex, 0, C);
		SimplexSet(simplex, 1, B);
		SimplexSet(simplex, 2, A);
		SimplexSetSize(simplex, 3);
	}

	return doSimplex3(simplex, dir);
}



#pragma endregion


__device__ float Vec3PointTriDist2(const float3 *P, const float3 *x0, const float3 *B, const float3 *C, float3 *witness)

{
	// Computation comes from analytic expression for triangle (x0, B, C)
	//      T(s, t) = x0 + s.d1 + t.d2, where d1 = B - x0 and d2 = C - x0 and
	// Then equation for distance is:
	//      D(s, t) = | T(s, t) - P |^2
	// This leads to minimization of quadratic function of two variables.
	// The solution from is taken only if s is between 0 and 1, t is
	// between 0 and 1 and t + s < 1, otherwise distance from segment is
	// computed.

	float3 d1, d2, a;
	float u, v, w, p, q, r, d;
	float s, t, dist, dist2;
	float3 witness2;

	float3Sub(&d1, B, x0);
	float3Sub(&d2, C, x0);
	float3Sub(&a, x0, P);

	u = float3Dot(&a, &a);
	v = float3Dot(&d1, &d1);
	w = float3Dot(&d2, &d2);
	p = float3Dot(&a, &d1);
	q = float3Dot(&a, &d2);
	r = float3Dot(&d1, &d2);

	d = w * v - r * r;
	if (IsZERO(d)) {
		// To avoid division by zero for zero (or near zero) area triangles
		s = t = -1.f;
	}
	else {
		s = (q * r - w * p) / d;
		t = (-s * r - q) / w;
	}

	if ((IsZERO(s) || s > ZERO)
		&& (floatEq(s, ONE) || s < ONE)
		&& (IsZERO(t) || t > ZERO)
		&& (floatEq(t, ONE) || t < ONE)
		&& (floatEq(t + s, ONE) || t + s < ONE)) {

		if (witness) {
			float3Scale(&d1, s);
			float3Scale(&d2, t);
			float3Copy(witness, x0);
			float3Add(witness, witness, &d1);
			float3Add(witness, witness, &d2);

			dist = float3Dist(witness, P);
		}
		else {
			dist = s * s * v;
			dist += t * t * w;
			dist += 2.f  * s * t * r;
			dist += 2.f * s * p;
			dist += 2.f  * t * q;
			dist += u;
		}
	}
	else {
		dist = PointSegmentDist(P, x0, B, witness);

		dist2 = PointSegmentDist(P, x0, C, &witness2);
		if (dist2 < dist) {
			dist = dist2;
			if (witness)
				float3Copy(witness, &witness2);
		}

		dist2 = PointSegmentDist(P, B, C, &witness2);
		if (dist2 < dist) {
			dist = dist2;
			if (witness)
				float3Copy(witness, &witness2);
		}
	}

	return dist;
}

__device__ float PointSegmentDist(const float3 *P, const float3 *x0, const float3 *b, float3 *witness)
{
	// The computation comes from solving equation of segment:
	//      S(t) = x0 + t.d
	//          where - x0 is initial point of segment
	//                - d is direction of segment from x0 (|d| > 0)
	//                - t belongs to <0, 1> interval
	// 
	// Than, distance from a segment to some point P can be expressed:
	//      D(t) = |x0 + t.d - P|^2
	//          which is distance from any point on segment. Minimization
	//          of this function brings distance from P to segment.
	// Minimization of D(t) leads to simple quadratic equation that's
	// solving is straightforward.
	//
	// Bonus of this method is witness point for free.

	float dist, t;
	float3 d, a;

	// direction of segment
	float3Sub(&d, b, x0);

	// precompute vector from P to x0
	float3Sub(&a, x0, P);

	t = -1.f * float3Dot(&a, &d);
	t /= float3Len(&d);

	if (t < ZERO || IsZERO(t)) {
		dist = float3Dist(x0, P);
		if (witness)
			float3Copy(witness, x0);
	}
	else if (t > ONE || floatEq(t, ONE)) {
		dist = float3Dist(b, P);
		if (witness)
			float3Copy(witness, b);
	}
	else {
		if (witness) {
			float3Copy(witness, &d);
			float3Scale(witness, t);
			float3Add(witness, witness, x0);
			dist = float3Dist(witness, P);
		}
		else {
			// recycling variables
			float3Scale(&d, t);
			float3Add(&d, &d, &a);
			dist = float3Len(&d);
		}
	}

	return dist;
}

__device__ void quatRotVec(float3 *v, const float4 *q)
{
	// original version: 31 mul + 21 add
	// optimized version: 18 mul + 12 add
	// formula: v = v + 2 * cross(q.xyz, cross(q.xyz, v) + q.w * v)
	float cross1_x, cross1_y, cross1_z, cross2_x, cross2_y, cross2_z;
	float x, y, z, w;
	float vx, vy, vz;

	vx = v->x;
	vy = v->y;
	vz = v->z;

	w = q->w;
	x = q->x;
	y = q->y;
	z = q->z;

	cross1_x = y * vz - z * vy + w * vx;
	cross1_y = z * vx - x * vz + w * vy;
	cross1_z = x * vy - y * vx + w * vz;
	cross2_x = y * cross1_z - z * cross1_y;
	cross2_y = z * cross1_x - x * cross1_z;
	cross2_z = x * cross1_y - y * cross1_x;
	*v = make_float3(vx + 2 * cross2_x, vy + 2 * cross2_y, vz + 2 * cross2_z);
}

__device__ int quatInvert2(float4 *dest, const float4 *src)
{
	float4Copy(dest, src);
	return quatInvert(dest);
}

__device__ int quatInvert(float4 *q)
{
	float len2 = magnitudeSqr(*q);
	if (len2 < FLT_EPSILON)
		return -1;

	len2 = ONE / len2;

	q->x = -q->x * len2;
	q->y = -q->y * len2;
	q->z = -q->z * len2;
	q->w = q->w * len2;

	return 0;
}

#pragma region  inlines

__device__ int Signum(float val)
{
	if (IsZERO(val)) {
		return 0;
	}
	else if (val < ZERO) {
		return -1;
	}
	return 1;
}


__device__ void tripleCross(const float3 *a, const float3 *b,
	const float3 *c, float3 *d)
{
	float3 e;
	float3Cross(&e, a, b);
	float3Cross(d, &e, c);
}


__device__ int IsZERO(float val)
{
	return absolute(val) < COL_EPS;
}

__device__ int floatEq(float a, float b)
{
	return a == b;
}

__device__ float absolute(float val)
{
	return val > 0 ? val : -val;
}

#pragma endregion



#pragma endregion
