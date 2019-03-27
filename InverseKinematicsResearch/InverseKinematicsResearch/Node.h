#pragma once
#include <vector>
#include "ik_constants.h"
#define GIZMO_SCALE_MATRIX glm::scale(glm::mat4(1.0f), glm::vec3(GIZMO_SIZE))


struct Connection;
class Node;
class OriginNode;
class EffectorNode;
class TargetNode;


struct Connection
{
public:
	Node* parent = nullptr;
	std::vector<Node*> children = std::vector<Node*>();
	float length = 0.0f;
};

class Node
{
#pragma region public 

public:
	
	glm::quat rotation;
	glm::quat minRotation;
	glm::quat maxRotation;

	Connection link;

	Node()
	{
		link.length = 0.0f;
		link.parent = nullptr;
		rotation = glm::quat(glm::vec3(0.0f));
	}

	Node(glm::vec3 rotation, glm::vec3 minRotation, glm::vec3 maxRotation, float length, Node* parent = nullptr)
	{
		link.length = length;
		link.parent = parent;
		

		this->rotation = glm::quat(rotation);
		this->minRotation = glm::quat(minRotation);
		this->maxRotation = glm::quat(maxRotation);
	}

	void AttachChild(Node* child)
	{
		child->link.parent = this;
		link.children.push_back(child);
	}

	void Draw(Shader shader, unsigned int VAO)
	{
		DrawCurrent(shader, VAO);
		for (int i = 0; i < link.children.size(); i++)
		{
			DrawLink(shader, VAO, link.children[i]);
			link.children[i]->Draw(shader, VAO);
		}
	}

	virtual glm::mat4 GetModelMatrix()
	{
		if (link.parent == nullptr)
		{
			return glm::mat4_cast(rotation);
		}
		else
		{
			return link.parent->GetModelMatrix() * glm::mat4_cast(rotation) * glm::translate(glm::mat4(1.0f), glm::vec3(link.length, 0.0f, 0.0f));
		}
	}

	void ToCUDA(NodeCUDA* allocatedPtr)
	{	
		int index = 0;
		CopyToArray(allocatedPtr, &index);
	}

	NodeCUDA* AllocateCUDA()
	{
		NodeCUDA* nodeC;
		int nodeCount = 1 + this->CountChildren();
		cudaMallocManaged(&nodeC, nodeCount * sizeof(NodeCUDA));
		return nodeC;
	}

	void FromCoords(CoordinatesNew coords, int *nodeIndex)
	{
		int coordIndex = (*nodeIndex - 1) * 3;
		this->rotation = glm::quat(glm::vec3(coords.positions[coordIndex], coords.positions[coordIndex + 1], coords.positions[coordIndex + 2]));
		(*nodeIndex)++;

		for (int i = 0; i < link.children.size(); i++)
		{
			link.children[i]->FromCoords(coords, nodeIndex);
		}
	}

#pragma endregion

#pragma region protected

protected:
	

	virtual void FillNodeCUDAtype(NodeCUDA* node)
	{
		node->nodeType = NodeType::node;
	}


	void CopyToArray(NodeCUDA* array, int* index, int parentIndex =-1)
	{

		NodeCUDA tmpNode = NodeCUDA();
		tmpNode.length = this->link.length;

		tmpNode.rotation.w= this->rotation.w;
		tmpNode.rotation.x=	this->rotation.x;
		tmpNode.rotation.y=	this->rotation.y;
		tmpNode.rotation.z=	this->rotation.z;	
		
		tmpNode.minRotation.w = this->minRotation.w;
		tmpNode.minRotation.x = this->minRotation.x;
		tmpNode.minRotation.y = this->minRotation.y;
		tmpNode.minRotation.z = this->minRotation.z;

		tmpNode.maxRotation.w = this->maxRotation.w;
		tmpNode.maxRotation.x = this->maxRotation.x;
		tmpNode.maxRotation.y = this->maxRotation.y;
		tmpNode.maxRotation.z = this->maxRotation.z;
			   
		tmpNode.parentIndex = parentIndex;
		this->FillNodeCUDAtype(&tmpNode);

		parentIndex = (*index);

		cudaMemcpy((void*)(array + *index), (void*)&tmpNode, sizeof(NodeCUDA), cudaMemcpyHostToDevice);

		(*index)++;

	
		for (int i = 0; i < link.children.size(); i++)
		{
			link.children[i]->CopyToArray(array, index ,parentIndex);
		}
		
	}


	virtual void DrawCurrent(Shader shader, unsigned int VAO)
	{
		shader.use();
		shader.setVec3("color", 0.0f, 1.0f, 0.0f);
		shader.setMat4("model", this->GetModelMatrix() * GIZMO_SCALE_MATRIX);
		glBindVertexArray(VAO);
		glLineWidth(2.0f);
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		glDrawArrays(GL_TRIANGLES, 0, 36);
	}

	void DrawLink(Shader shader, unsigned int VAO, Node* child)
	{
		glm::mat4 model = GetModelMatrix() *
						  glm::mat4_cast(child->rotation)*
						  glm::translate(glm::mat4(1.0f), glm::vec3(child->link.length * 0.5f, 0.0f, 0.0f)) *
						  glm::scale(glm::mat4(1.0f), glm::vec3(child->link.length, GIZMO_SIZE * 0.25f, GIZMO_SIZE * 0.25f));

		shader.use();
		shader.setVec3("color", 1.0f, 0.5f, 0.2f);
		shader.setMat4("model", model);
		glBindVertexArray(VAO);
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		glDrawArrays(GL_TRIANGLES, 0, 36);
	}

#pragma endregion

private:

	

	int CountChildren()
	{
		int sum = 0;
		
		if (link.children.size()!=0)
		{
			for (int i = 0; i < link.children.size(); i++)
			{
				sum += link.children[i]->CountChildren()+1;
			}
		}
		
		return sum;

	}

};

class OriginNode : public Node
{

public:

	glm::vec3 position;
	

	OriginNode(glm::vec3 position = glm::vec3(0.0f),
			   glm::vec3 rotation = glm::vec3(0.0f), 
			   glm::vec3 minRotation = glm::vec3(-1.0f*PI, -1.0f*PI, -1.0f*PI),
			   glm::vec3 maxRotation = glm::vec3(+1.0f*PI, +1.0f*PI, +1.0f*PI)):Node()
	{
		this->position = position;
		this->rotation = glm::quat(rotation);
		this->minRotation = glm::quat(minRotation);
		this->maxRotation = glm::quat(maxRotation);
	}
	
	glm::mat4 GetModelMatrix() override
	{
		return glm::translate(glm::mat4(1.0f), this->position) * glm::mat4_cast(rotation);
	}

protected:

	void FillNodeCUDAtype(NodeCUDA* node) override
	{
		node->nodeType = NodeType::originNode;

		node->position.x = this->position.x;
		node->position.y = this->position.y;
		node->position.z = this->position.z;

	}

	virtual void DrawCurrent(Shader shader, unsigned int VAO)
	{
		shader.use();
		shader.setVec3("color", 1.0f, 1.0f, 1.0f);
		shader.setMat4("model", this->GetModelMatrix() * GIZMO_SCALE_MATRIX);
		glBindVertexArray(VAO);
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		glDrawArrays(GL_TRIANGLES, 0, 36);
		shader.setVec3("color", 0.0f, 0.0f, 0.0f);
		glLineWidth(2.0f);
		glDrawArrays(GL_LINE_STRIP, 0, 36);
	}
};

class TargetNode
{
public:
	glm::vec3 position;
	glm::quat rotation;

	glm::mat4 GetModelMatrix()
	{
		return glm::translate(glm::mat4(1.0f), position) * glm::mat4_cast(rotation);
	}

	TargetNode(glm::vec3 position = glm::vec3(0.0f), glm::vec3 rotation = glm::vec3(0.0f))
	{
		this->position = position;
		this->rotation = glm::quat(rotation);
	}

	void DrawCurrent(Shader shader, unsigned int VAO)
	{
		shader.use();
		shader.setVec3("color", 1.0f, 0.0f, 0.0f);
		shader.setMat4("model", this->GetModelMatrix() * GIZMO_SCALE_MATRIX);
		glBindVertexArray(VAO);
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		glDrawArrays(GL_TRIANGLES, 0, 36);
	}

	void translate(glm::vec3 translation)
	{
		position += translation;
	}

};


class EffectorNode : public Node
{
public:
	TargetNode* target;

	EffectorNode(glm::vec3 rotation, glm::vec3 minRotation, glm::vec3 maxRotation, float length, TargetNode* target = nullptr, Node* parent = nullptr) : Node(rotation,minRotation,maxRotation,length, parent)
	{
		this->target = target;
	}

protected:

	void FillNodeCUDAtype(NodeCUDA* node) override
	{
		node->nodeType = NodeType::effectorNode;
		if (target != nullptr)
		{
			node->targetPosition.x = this->target->position.x;
			node->targetPosition.y = this->target->position.y;
			node->targetPosition.z = this->target->position.z;

			node->targetRotation.w = this->target->rotation.w;
			node->targetRotation.x = this->target->rotation.x;
			node->targetRotation.y = this->target->rotation.y;
			node->targetRotation.z = this->target->rotation.z;	

		}
	}

	virtual void DrawCurrent(Shader shader, unsigned int VAO)
	{
		shader.use();
		shader.setVec3("color", 1.0f, 1.0f, 0.0f);
		shader.setMat4("model", this->GetModelMatrix() * GIZMO_SCALE_MATRIX);
		glBindVertexArray(VAO);
		glLineWidth(2.0f);
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		glDrawArrays(GL_TRIANGLES, 0, 36);
	}
};

