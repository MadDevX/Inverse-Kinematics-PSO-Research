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

struct KinematicChainCuda
{
	float3 _shoulderPosition;
	float3 _shoulderRotation;
	float3 _elbowRotation;
	float3 _minShoulder;
	float3 _maxShoulder;
	float3 _minElbow;
	float3 _maxElbow;
	float _armLength;
	float _forearmLength;
};

struct Coordinates
{
	float shoulderRotX = 0.0f;
	float shoulderRotY = 0.0f;
	float shoulderRotZ = 0.0f;
	float elbowRotX = 0.0f;
	float elbowRotY = 0.0f;
	float elbowRotZ = 0.0f;
};

struct CoordinatesNew
{
	float *positions;

	CoordinatesNew()
	{
		cudaMallocManaged((void**)&positions, DEGREES_OF_FREEDOM * sizeof(float));
		for (int i = 0; i < DEGREES_OF_FREEDOM; i++)
		{
			positions[i] = 0.0f;
		}
	}

	CoordinatesNew(const CoordinatesNew &coords)
	{
		cudaMallocManaged((void**)&positions, DEGREES_OF_FREEDOM * sizeof(float));
		for (int i = 0; i < DEGREES_OF_FREEDOM; i++)
		{
			positions[i] = coords.positions[i];
		}
	}

	CoordinatesNew& operator=(const CoordinatesNew &coords)
	{
		for (int i = 0; i < DEGREES_OF_FREEDOM; i++)
		{
			positions[i] = coords.positions[i];
		}
		return *this;
	}

	~CoordinatesNew()
	{
		cudaFree(positions);
	}
};

struct ParticleNew
{
	float positions[DEGREES_OF_FREEDOM];
	float velocities[DEGREES_OF_FREEDOM];
	float localBest[DEGREES_OF_FREEDOM];
};

struct Particle
{
	Coordinates positions;
	Coordinates velocities;
	Coordinates localBest;
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