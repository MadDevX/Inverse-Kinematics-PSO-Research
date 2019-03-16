#pragma once
#include <vector>
#define GIZMO_SIZE 0.2f
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
public:
	glm::quat rotation;
	Connection link;

	Node()
	{
		link.length = 0.0f;
		link.parent = nullptr;
		rotation = glm::quat(glm::vec3(0.0f));
	}

	Node(glm::vec3 rotation, float length, Node* parent = nullptr)
	{
		link.length = length;
		link.parent = parent;
		this->rotation = glm::quat(rotation);
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
			return link.parent->GetModelMatrix()* glm::mat4_cast(rotation) * glm::translate(glm::mat4(1.0f), glm::vec3(link.length, 0.0f, 0.0f)) ;
		}
	}

	//NodeCUDA* ToCUDA()
	//{
	//	unsigned int size = 0;
	//	for (int i = 0; i < link.children.size(); i++)
	//	{
	//		size += sizeof(*link.children[i]);
	//	}
	//}

protected:
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
};

class OriginNode : public Node
{
public:
	glm::vec3 position;

	OriginNode(glm::vec3 position = glm::vec3(0.0f), glm::vec3 rotation = glm::vec3(0.0f)):Node()
	{
		this->position = position;
		this->rotation = glm::quat(rotation);
	}
	
	glm::mat4 GetModelMatrix() override
	{
		return glm::translate(glm::mat4(1.0f), this->position) * glm::mat4_cast(rotation);
	}

protected:
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

class EffectorNode : public Node
{
public:
	TargetNode* target;

	EffectorNode(glm::vec3 rotation, float length, TargetNode* target = nullptr, Node* parent = nullptr) : Node(rotation, length, parent)
	{
		this->target = target;
	}

protected:
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

class TargetNode
{
public:
	glm::vec3 position;
	glm::quat rotation;

	glm::mat4 GetModelMatrix()
	{
		return glm::translate(glm::mat4(1.0f), position) * glm::mat4_cast(rotation);
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
};