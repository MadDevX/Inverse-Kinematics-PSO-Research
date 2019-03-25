#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include "ik_constants.h"

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
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