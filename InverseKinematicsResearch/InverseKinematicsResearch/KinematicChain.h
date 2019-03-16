#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "Particle.h"

glm::mat4 rotateEuler(glm::mat4 matrix, glm::vec3 angles)
{
	matrix = glm::rotate(matrix, angles.x, glm::vec3(1.0f, 0.0f, 0.0f));
	matrix = glm::rotate(matrix, angles.y, glm::vec3(0.0f, 1.0f, 0.0f));
	matrix = glm::rotate(matrix, angles.z, glm::vec3(0.0f, 0.0f, 1.0f));
	return matrix;
}

glm::vec3 clampVec3(glm::vec3 vector, glm::vec3 min, glm::vec3 max)
{
	vector.x = glm::min(vector.x, max.x);
	vector.x = glm::max(vector.x, min.x);
	vector.y = glm::min(vector.y, max.y);
	vector.y = glm::max(vector.y, min.y);
	vector.z = glm::min(vector.z, max.z);
	vector.z = glm::max(vector.z, min.z);
	return vector;
}

float3 fromGLM(glm::vec3 vector)
{
	return make_float3(vector.x, vector.y, vector.z);
}

class KinematicChain
{
private:
	float PI = glm::pi<float>();

public:
	glm::vec3 _minElbow = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 _maxElbow = glm::vec3(0.0f, PI * 0.8f, 0.0f);
	glm::vec3 _minShoulder = glm::vec3(PI * (-0.5f), PI * (-0.25f), PI * (-0.5f));
	glm::vec3 _maxShoulder = glm::vec3(PI * 0.5f, PI * 0.8f, PI * 0.5f);

	glm::vec3 _shoulderPosition;
	glm::vec3 _shoulderRotation;
	glm::vec3 _elbowRotation;
	glm::vec3 _wristRotation;

	float _armLength;
	float _forearmLength;
	float _gizmoScale;

	KinematicChain(glm::vec3 shoulderPosition, float armLength, float forearmLength, float scale,
				   glm::vec3 shoulderRotation = glm::vec3(0.0f, 0.0f, 0.0f), 
				   glm::vec3 elbowRotation = glm::vec3(0.0f, 0.0f, 0.0f), 
				   glm::vec3 wristRotation = glm::vec3(0.0f, 0.0f, 0.0f))
	{
		_shoulderPosition = shoulderPosition;
		_elbowRotation = elbowRotation;
		_armLength = armLength;
		_forearmLength = forearmLength;
		_shoulderRotation = shoulderRotation;
		_gizmoScale = scale;
	}

	glm::mat4 getShoulderMatrix()
	{
		glm::mat4 model = glm::mat4(1.0f);
		model = glm::translate(model, _shoulderPosition);
		model = rotateEuler(model, _shoulderRotation);
		model = glm::scale(model, glm::vec3(_gizmoScale));

		return model;
	}

	glm::mat4 getElbowMatrix()
	{
		glm::mat4 model = getShoulderMatrix();
		model = glm::translate(model, glm::vec3(_armLength / _gizmoScale, 0.0f, 0.0f));
		model = rotateEuler(model, _elbowRotation);

		return model;
	}

	glm::mat4 getWristMatrix()
	{
		glm::mat4 model = getElbowMatrix();
		model = glm::translate(model, glm::vec3(_forearmLength / _gizmoScale, 0.0f, 0.0f));
		model = rotateEuler(model, _wristRotation);

		return model;
	}

	void translateShoulder(glm::vec3 translation)
	{
		_shoulderPosition += translation;
	}

	void rotateElbow(glm::vec3 rotation)
	{
		_elbowRotation += rotation;
		_elbowRotation = clampVec3(_elbowRotation, _minElbow, _maxElbow);
	}

	void rotateShoulder(glm::vec3 rotation)
	{
		_shoulderRotation += rotation;
		_shoulderRotation = clampVec3(_shoulderRotation, _minShoulder, _maxShoulder);
	}

	KinematicChainCuda toCuda()
	{
		KinematicChainCuda c;
		c._armLength =        _armLength;
		c._forearmLength =    _forearmLength;
		c._shoulderPosition = fromGLM(_shoulderPosition);
		c._shoulderRotation = fromGLM(_shoulderRotation);
		c._elbowRotation =    fromGLM(_elbowRotation);
		c._minShoulder =      fromGLM(_minShoulder);
		c._maxShoulder =      fromGLM(_maxShoulder);
		c._minElbow =         fromGLM(_minElbow);
		c._maxElbow =         fromGLM(_maxElbow);

		return c;
	}

	void fromCoords(Coordinates coords)
	{
		_shoulderRotation.x = coords.shoulderRotX;
		_shoulderRotation.y = coords.shoulderRotY;
		_shoulderRotation.z = coords.shoulderRotZ;
		_elbowRotation.x = coords.elbowRotX;
		_elbowRotation.y = coords.elbowRotY;
		_elbowRotation.z = coords.elbowRotZ;
	}
};

class Target
{
public:
	glm::vec3 _position;
	glm::vec3 _rotation;
	float _gizmoScale;

	Target(glm::vec3 position, glm::vec3 rotation, float scale)
	{
		_position = position;
		_rotation = rotation;
		_gizmoScale = scale;
	}

	glm::mat4 getModelMatrix()
	{
		glm::mat4 model(1.0f);
		model = glm::translate(model, _position);
		model = rotateEuler(model, _rotation);
		model = glm::scale(model, glm::vec3(_gizmoScale));

		return model;
	}

	void translate(glm::vec3 translation)
	{
		_position += translation;
	}
};