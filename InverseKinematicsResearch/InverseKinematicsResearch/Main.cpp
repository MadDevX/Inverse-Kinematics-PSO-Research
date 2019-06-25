#include <stdio.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <LearnOpenGL/Shader.h>
#include <im3d/im3d.h>
#include <cuda.h>
#include <curand_kernel.h>
#include "Models.h"
#include "Particle.h"
#include "Node.h"
#include "BoxCollider.h"
int WINDOW_WIDTH = 800;
int WINDOW_HEIGHT = 600;
int N = 8192;
int colliderCount = 4;
float deltaTime = 0.0f;
float lastFrame = 0.0f;

double lastX = 0.0f, lastY = 0.0f;
float rotationY = 0.0f;
bool rotate = false;

glm::mat4 view = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -5.0f));

extern cudaError_t initGenerators(curandState_t *randoms, int size);
extern cudaError_t calculatePSO(float *particles,float* positions, float *bests, curandState_t *randoms, int size, NodeCUDA *chain, PSOConfig PSOconfig,FitnessConfig fitnessConfig, Coordinates *result, obj_t* colliders, int colliderCount);
GLFWwindow* initOpenGLContext();
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void processInput(GLFWwindow *window);
void calculateDeltaTime();
unsigned int initVAO(unsigned int *VBO);
unsigned int initCoordVAO(unsigned int *VBO);
void drawCoordinates(Shader shader, unsigned int VAO);
void initColliders(obj_t* colliders, int colliderCount);
void drawColliders(obj_t* colliders, int colliderCount, Shader shader, unsigned int VAO);
void rotateCollider(obj_t* collider, float time);

std::ofstream openStream(char* fileName);
void writeData(std::ofstream stream, char* data);
void closeStr(std::ofstream stream);

void fillPositionData(float* positions, Node * Origin,int*ind);


TargetNode* movingTarget;
OriginNode* nodeArm;
TargetNode** targets;


int main(int argc, char** argv)
{
	if (argc == 2)
	{
		N = atoi(argv[1]);
	}

	#pragma region GLfunctions

	GLFWwindow* window = initOpenGLContext();
	Shader shader("3.3.jointShader.vert", "3.3.jointShader.frag");
	unsigned int VBO, coordVBO, linkVBO;
	unsigned int VAO = initVAO(&VBO);
	unsigned int coordVAO = initCoordVAO(&coordVBO);

	#pragma endregion

	#pragma region Arm setup
	nodeArm = new OriginNode(glm::vec3(0.0f), glm::vec3(0.0f), glm::vec3(0.0f), glm::vec3(2 * PI));
	int elbows = 4;
	Node** nodeElbows = new Node*[elbows];
	for (int i = 0; i < elbows; i++)
	{
		nodeElbows[i] = new Node(glm::vec3(0.0f, 1.57f, 0.0f), glm::vec3(0.0f), glm::vec3(2 * PI), 1.0f);
	}
	EffectorNode* nodeWrist = new EffectorNode(1.0f,glm::vec3(0.0f, 1.57f, 0.0f), glm::vec3(0.0f), glm::vec3(2 * PI), 1.0f);
	EffectorNode* nodeWrist2 = new EffectorNode(1.0f, glm::vec3(0.0f, 0.0f, 1.57f), glm::vec3(0.0f), glm::vec3(2 * PI), 1.0f);
	EffectorNode* nodeWrist3 = new EffectorNode(1.0f, glm::vec3(0.0f, 0.0f, 1.57f), glm::vec3(0.0f), glm::vec3(2 * PI), 1.0f);
	TargetNode* nodeTarget1 = new TargetNode(glm::vec3(1.0f, 1.0f, -1.5f));
	TargetNode* nodeTarget2 = new TargetNode(glm::vec3(-1.0f, 1.0f, -1.5f));
	TargetNode* nodeTarget3 = new TargetNode(glm::vec3(0.0f, 0.0f, -2.0f));

	movingTarget = nodeTarget1;
	targets = (TargetNode**)malloc(AMOUNT_OF_TARGETS * sizeof(TargetNode*));

	targets[0] = nodeTarget1;
	targets[1] = nodeTarget2;
	targets[2] = nodeTarget3;

	nodeWrist->target = nodeTarget1;
	nodeWrist2->target = nodeTarget2;
	nodeWrist3->target = nodeTarget3;



	nodeArm->AttachChild(nodeElbows[0]);
	for (int i = 1; i < elbows; i++)
	{
		nodeElbows[i - 1]->AttachChild(nodeElbows[i]);
	}
	nodeElbows[elbows - 1]->AttachChild(nodeWrist);
	nodeElbows[elbows - 1]->AttachChild(nodeWrist2);
	nodeElbows[elbows - 1]->AttachChild(nodeWrist3);

#pragma endregion

   
	NodeCUDA* chainCuda = nodeArm->AllocateCUDA();
	float *armPositions = nodeArm->AllocatePositions();

	curandState_t *randoms;
	float *particles;
	obj_t* colliders;
	Coordinates* resultCoords;

	PSOConfig psoConfig(0.5f, 0.5f, 1.25f, 15);
	FitnessConfig fitConfig(3.0f,0.5f,0.1f);
	float *bests;

	cudaMalloc((void**)&randoms, N * sizeof(curandState_t));
	cudaMalloc((void**)&particles, N * 3 * DEGREES_OF_FREEDOM * sizeof(float));
	cudaMalloc((void**)&bests, N * sizeof(float));
	cudaMallocManaged((void**)&colliders, colliderCount * sizeof(obj_t));
	cudaMallocManaged((void**)&resultCoords, sizeof(Coordinates));
	initColliders(colliders, colliderCount);
	initGenerators(randoms, N);

	std::ofstream posStream;
	std::ofstream degStream;

	posStream = openStream("IK-diagnostics-positions.txt");
	degStream = openStream("IK-diagnostics-degrees.txt");
	
	char* data = "3.34234234\0";

	float * posArray = (float*)malloc( (NODE_COUNT - 1) * sizeof(float) * 3 );
	float * degsArray;


	while (!glfwWindowShouldClose(window))
	{

		calculateDeltaTime();
		processInput(window);
		glfwPollEvents();

		//rotateCollider(&colliders[0], (float)glfwGetTime());

		cudaError_t status;

		nodeArm->ToCUDA(chainCuda);
		nodeArm->FillPositions(armPositions, chainCuda); 

#pragma region write to files
		int k = 0;
		fillPositionData(posArray, nodeArm, &k);
		degsArray = resultCoords->positions;

		for (int deg = 0; deg < DEGREES_OF_FREEDOM; deg++)
		{
			degStream << degsArray[deg] << ";";
		}
		degStream << "\n";
		
		for (int deg = 0; deg < DEGREES_OF_FREEDOM; deg++)
		{
			posStream << posArray[deg] << ";";
		}
		posStream << "\n";

#pragma endregion



		status = calculatePSO(particles,armPositions, bests, randoms, N, chainCuda, psoConfig, fitConfig, resultCoords, colliders, colliderCount);
		if (status != cudaSuccess) break;
		int ind = 0;

		nodeArm->FromCoords(resultCoords, &ind);
		#pragma region GLrendering

		glClearColor(0.3f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		
		shader.use();
		shader.setMat4("view", view);
		shader.setMat4("projection", glm::perspective(glm::radians(45.0f), (float)WINDOW_WIDTH / WINDOW_HEIGHT, 0.1f, 100.0f));
		drawCoordinates(shader, coordVAO);
		nodeArm->Draw(shader, VAO);
		nodeTarget1->DrawCurrent(shader, VAO);
		nodeTarget2->DrawCurrent(shader, VAO);
		nodeTarget3->DrawCurrent(shader, VAO);
		drawColliders(colliders, colliderCount, shader, VAO);

		glfwSwapBuffers(window);
		
		#pragma endregion

	}
	
	#pragma region cleanup
	glDeleteVertexArrays(1, &VAO);
	glDeleteVertexArrays(1, &coordVAO);
	glDeleteBuffers(1, &VBO);
	glDeleteBuffers(1, &coordVBO);

	glfwTerminate();

	posStream.close();
	degStream.close();

	cudaFree(armPositions);
	cudaFree(bests);
	cudaFree(chainCuda);
	cudaFree(particles);
	cudaFree(randoms);
	cudaFree(colliders);
	cudaFree(resultCoords);
	delete(nodeArm);
	
	for (int i = 0; i < elbows; i++)
	{
		delete(nodeElbows[i]);
	}

	delete[](nodeElbows);
	delete(nodeWrist);
	delete(nodeWrist2);
	delete(nodeWrist3);
	#pragma endregion

	
	return 0;
}

#pragma region TextSaving

std::ofstream openStream(char * filename)
{
	std::ofstream ofs(filename, std::ofstream::out|std::ofstream::app);
	return ofs;
}


void writeData(std::ofstream stream, char* data)
{
	//fwrite(data, sizeof(char), sizeof(data), file);
	stream << data;
}


void closeStream(std::ofstream stream)
{
	stream.close();
}

#pragma endregion

void fillPositionData(float * positions, Node * Origin, int* ind)
{
	int index = (*ind) -1;
	if (index != -1)
	{
		
		index *= 3;
		glm::vec4 originVector(0.0f, 0.0f, 0.0f, 1.0f);
		glm::mat4 model = Origin->GetModelMatrix();
		glm::vec4 position = model * originVector;

		positions[index] = position.x;
		positions[index+1] = position.y;
		positions[index+2] = position.z;
	}
	
	for (int i = 0; i < Origin->link.children.size(); i++)
	{
		*ind = *ind + 1;
		fillPositionData(positions, Origin->link.children[i], ind);
	}

}


#pragma region GLfun



unsigned int initVAO(unsigned int *VBO)
{
	unsigned int VAO;
	glGenVertexArrays(1, &VAO);
	
	glGenBuffers(1, VBO);

	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, *VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(VERTICES_CUBE), VERTICES_CUBE, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	return VAO;
}

unsigned int initCoordVAO(unsigned int *VBO)
{
	unsigned int VAO;
	glGenVertexArrays(1, &VAO);

	glGenBuffers(1, VBO);

	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, *VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(VERTICES_GRID), VERTICES_GRID, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	return VAO;
}



void drawCoordinates(Shader shader, unsigned int VAO)
{
	shader.use();
	glBindVertexArray(VAO);
	glLineWidth(2.0f);
	shader.setMat4("view", view);
	shader.setMat4("model", glm::mat4(1.0f));
	shader.setVec3("color", 1.0f, 0.0f, 0.0f);
	glDrawArrays(GL_LINES, 0, 2);
	shader.setVec3("color", 0.0f, 1.0f, 0.0f);
	glDrawArrays(GL_LINES, 2, 2);
	shader.setVec3("color", 0.0f, 0.0f, 1.0f);
	glDrawArrays(GL_LINES, 4, 2);
}


void configureGLFWContext()
{
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
}

void processInput(GLFWwindow *window)
{

	for (int i = 0; i < AMOUNT_OF_TARGETS; i++)
	{
		if (glfwGetKey(window, GLFW_KEY_1+i) == GLFW_PRESS)
			movingTarget = (targets[i]);
	}

	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
	if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
		nodeArm->translate(glm::vec3(-1.0f, 0.0f, 0.0f) * deltaTime);
	if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
		nodeArm->translate(glm::vec3(1.0f, 0.0f, 0.0f) * deltaTime);
	if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
		nodeArm->translate(glm::vec3(0.0f, 1.0f, 0.0f) * deltaTime);
	if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
		nodeArm->translate(glm::vec3(0.0f, -1.0f, 0.0f) * deltaTime);
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		movingTarget->translate(glm::vec3(-1.0f, 0.0f, 0.0f) * deltaTime);
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		movingTarget->translate(glm::vec3(1.0f, 0.0f, 0.0f) * deltaTime);
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		movingTarget->translate(glm::vec3(0.0f, 1.0f, 0.0f) * deltaTime);
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		movingTarget->translate(glm::vec3(0.0f, -1.0f, 0.0f) * deltaTime);
	if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
		movingTarget->translate(glm::vec3(0.0f, 0.0f, 1.0f) * deltaTime);
	if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
		movingTarget->translate(glm::vec3(0.0f, 0.0f, -1.0f) * deltaTime);
	if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS)
	{
		rotate = true;
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	}
	else
	{
		rotate = false;
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
	}
}

GLFWwindow* createWindow()
{
	GLFWwindow* window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Inverse Kinematics PSO", NULL, NULL);
	if (window == NULL)
	{
		glfwTerminate();
		throw - 1;
	}
	glfwMakeContextCurrent(window);

	return window;
}

GLFWwindow* initOpenGLContext()
{
	glfwInit();
	configureGLFWContext();

	GLFWwindow* window;

	try
	{
		window = createWindow();
	}
	catch (int)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		exit(EXIT_FAILURE);
	}

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		exit(EXIT_FAILURE);
	}

	glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

	glEnable(GL_DEPTH_TEST);
	glfwSetCursorPosCallback(window, mouse_callback);

	return window;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	height = height == 0 ? 1: height;
	width = width == 0 ? 1 : width;
	glViewport(0, 0, width, height);
	WINDOW_WIDTH = width;
	WINDOW_HEIGHT = height;
}

void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{

	float xoffset = (float)xpos - lastX;
	float yoffset = lastY - (float)ypos;

	lastX = (float)xpos;
	lastY = (float)ypos;

	if(rotate)
	{
		rotationY += xoffset * 0.01f;
		glm::mat4 rotMat = glm::rotate(glm::mat4(1.0f), xoffset * 0.01f, glm::vec3(0.0f, 1.0f, 0.0f));
		view = view * rotMat;
		rotMat = glm::rotate(glm::mat4(1.0f), -yoffset * 0.01f, glm::vec3(cosf(rotationY), 0.0f, sinf(rotationY)));
		view = view * rotMat;

	}
	
}

void calculateDeltaTime()
{
	float currentFrame = (float)glfwGetTime();
	deltaTime = currentFrame - lastFrame;
	lastFrame = currentFrame;
}

void initColliders(obj_t* colliders, int colliderCount)
{
	colliders[0].pos = make_float3(1.0f, 0.0f, 0.0f);
	colliders[0].quat = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
	colliders[0].x = colliders[0].y = colliders[0].z = 1.0f;

	colliders[1].pos = make_float3(0.0f, 0.0f, -1.0f);
	colliders[1].quat = make_float4(-0.403f, -0.819f, 0.273f, 0.304f);
	colliders[1].x = colliders[1].y = colliders[1].z = 1.0f;

	colliders[2].pos = make_float3(-1.0f, 0.0f, 0.0f);
	colliders[2].quat = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
	colliders[2].x = colliders[2].y = colliders[2].z = 1.0f;

	colliders[3].pos = make_float3(0.0f, 0.0f, 1.0f);
	colliders[3].quat = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
	colliders[3].x = colliders[3].y = colliders[3].z = 1.0f;
}

void rotateCollider(obj_t* collider, float time)
{
	glm::quat rotation(glm::vec3(time, time, time));
	collider->quat.x = rotation.x;
	collider->quat.y = rotation.y;
	collider->quat.z = rotation.z;
	collider->quat.w = rotation.w;
}

void drawBoxCollider(obj* collider, Shader shader, unsigned int VAO)
{
	shader.use();
	shader.setVec3("color", 0.714f, 0.298f, 0.035f);
	glm::mat4 model(1.0f);
	model = glm::translate(model, glm::vec3(collider->pos.x, collider->pos.y, collider->pos.z)) * 
		    glm::mat4_cast(glm::quat(collider->quat.w, collider->quat.x, collider->quat.y, collider->quat.z)) * 
		    glm::scale(model, glm::vec3(collider->x, collider->y, collider->z));
	shader.setMat4("model", model);
	glBindVertexArray(VAO);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glDrawArrays(GL_TRIANGLES, 0, 36);
	shader.setVec3("color", 0.0f, 0.0f, 0.0f);
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glLineWidth(2.0f);
	glDrawArrays(GL_LINE_STRIP, 0, 36);
}

void drawColliders(obj_t* colliders, int colliderCount, Shader shader, unsigned int VAO)
{
	for (int i = 0; i < colliderCount; i++)
	{
		drawBoxCollider(&colliders[i], shader, VAO);
	}
}
#pragma endregion