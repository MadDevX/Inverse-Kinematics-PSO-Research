#pragma once
#include <stdio.h>
#include <vector_types.h>

struct NodeCUDA
{
	int nodeType;
	int parentIndex;
	float3 position;
	float4 rotation;
	float length;
	float3 targetPosition;
	float4 targetRotation;
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


	Config(float inertia = 0.2f, float local = 0.7f, float global = 0.5f, int iterations = 15)
	{
		_inertia = inertia;
		_local = local;
		_global = global;
		_iterations = iterations;
	}
};