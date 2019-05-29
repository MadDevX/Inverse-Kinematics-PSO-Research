#pragma once
#include <stdio.h>
#include <vector_types.h>
#include "ik_constants.h"

struct Matrix
{
	float cells[16];
};

enum NodeType
{
	originNode,
	effectorNode,
	node
};

struct NodeCUDA
{
	NodeType nodeType;
	int parentIndex;
	float3 position;
	float3 rotation; //rotation relative to parent

	float3 maxRotation;
	float3 minRotation;

	float length;
	float3 targetPosition;
	float3 targetRotation;
};

struct Coordinates
{
	float *positions;

	Coordinates()
	{
		cudaMallocManaged((void**)&positions, DEGREES_OF_FREEDOM * sizeof(float));
		for (int i = 0; i < DEGREES_OF_FREEDOM; i++)
		{
			positions[i] = 0.0f;
		}
	}

	Coordinates(const Coordinates &coords)
	{
		cudaMallocManaged((void**)&positions, DEGREES_OF_FREEDOM * sizeof(float));
		for (int i = 0; i < DEGREES_OF_FREEDOM; i++)
		{
			positions[i] = coords.positions[i];
		}
	}

	Coordinates& operator=(const Coordinates &coords)
	{
		for (int i = 0; i < DEGREES_OF_FREEDOM; i++)
		{
			positions[i] = coords.positions[i];
		}
		return *this;
	}

	~Coordinates()
	{
		cudaFree(positions);
	}
};

struct Particle
{
	float positions[DEGREES_OF_FREEDOM];
	float velocities[DEGREES_OF_FREEDOM];
	float localBest[DEGREES_OF_FREEDOM];
};

struct Config
{
	float _inertia;
	float _local;
	float _global;
	int _iterations;


	Config(float inertia = 0.2f, float local = 0.5f, float global = 0.7f, int iterations = 10)
	{
		_inertia = inertia;
		_local = local;
		_global = global;
		_iterations = iterations;
	}
};