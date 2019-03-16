#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/quaternion.hpp>
#include "Particle.h"

struct Connection
{
	Node* parent;
	float length;
};

class Node
{
public:
	glm::quat rotation;
	Connection link;

	Node()
	{
		link.length = 0.0f;
		link.parent = nullptr;
		rotation = glm::quat(glm::vec3(0.0f));
	}

	Node(Node* parent, glm::vec3 rotation, float length)
	{
		link.length = length;
		link.parent = parent;
		this->rotation = glm::quat(rotation);
	}

	glm::mat4 virtual GetModelMatrix()
	{
		if (link.parent == nullptr)
		{
			return glm::mat4_cast(rotation);
		}
		else
		{
			return link.parent->GetModelMatrix() * glm::translate(glm::mat4_cast(rotation), glm::vec3(link.length, 0.0f, 0.0f));
		}
	}
};

class OriginNode : Node
{
public:
	glm::vec3 position;

	OriginNode(glm::vec3 position, glm::vec3 rotation):Node()
	{
		this->position = position;
		this->rotation = glm::quat(rotation);
	}
	
	glm::mat4 GetModelMatrix() override
	{
		return glm::translate(glm::mat4(1.0f), this->position) * glm::mat4_cast(rotation);
	}
};

class EffectorNode : Node
{
public:
	Target* target;
};

class Target
{
public:
	glm::vec3 position;
	glm::quat rotation;
};